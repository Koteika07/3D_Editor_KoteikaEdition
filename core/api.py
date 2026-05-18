"""API для работы со сценой"""

from pathlib import Path
from typing import List, Tuple
import numpy as np

from geometry import EPSILON, as_vector3, Point, Edge, Face, Polyhedron, create_cube, create_pyramid, create_sphere, create_sphere_quads, create_icosahedron
from core import Scene, ObjectType

class GeometryAPI:
    def __init__(self):
        self.scene = Scene()

    def _next_id(self, obj_type: ObjectType) -> int:
        """Генерация следующего ID"""
        return self.scene._reserve_id(obj_type)

    def add_point(self, position: np.ndarray) -> int:
        p_id = self._next_id(ObjectType.POINT)
        self.scene.points[p_id] = Point(p_id, position)
        return p_id

    def add_edge(self, p1: int, p2: int) -> int:
        e_id = self._next_id(ObjectType.EDGE)
        self.scene.edges[e_id] = Edge(e_id, p1, p2)
        return e_id

    def add_face(self, vertex_ids: List[int], plane_id: int = -1) -> int:
        f_id = self._next_id(ObjectType.FACE)

        edge_ids = []
        for i in range(len(vertex_ids)):
            v1, v2 = vertex_ids[i], vertex_ids[(i+1) % len(vertex_ids)]
            edge_ids.append(self.add_edge(v1, v2))

        self.scene.faces[f_id] = Face(f_id, vertex_ids, plane_id, edge_ids)
        return f_id

    def add_polyhedron(self, face_ids: List[int], name: str = "") -> int:
        """Создаёт многогранник из существующих граней"""
        pl_id = self._next_id(ObjectType.POLYHEDRON)
        self.scene.polyhedra[pl_id] = Polyhedron(pl_id, face_ids, name)
        return pl_id

    def _create_from_factory(self, factory_func, name: str, **kwargs) -> int:
        """Общий метод для создания примитивов по фабрике"""
        vertices, faces_idx = factory_func(**kwargs)

        vertex_ids = [self.add_point(v) for v in vertices]

        face_ids = []
        for face_vertices in faces_idx:
            global_vids = [vertex_ids[i] for i in face_vertices]
            face_ids.append(self.add_face(global_vids))

        return self.add_polyhedron(face_ids, name)

    def create_cube(self, center=(0,0,0), size=1.0, name="cube") -> int:
        """Создаёт куб и возвращает ID"""
        return self._create_from_factory(
            create_cube, name, 
            center=center, size=size
        )

    def create_pyramid(self, center=(0,0,0), base_size=1.0, height=1.0, name="pyramid") -> int:
        """Создаёт пирамиду и возвращает ID"""
        return self._create_from_factory(
            create_pyramid, name,
            center=center, base_size=base_size, height=height
        )

    def create_sphere(self, center=(0,0,0), radius=1.0, segments=12, rings=6, name="sphere") -> int:
        """Создаёт сферу и возвращает ID"""
        return self._create_from_factory(
            create_sphere, name,
            center=center, radius=radius, segments=segments, rings=rings
        )

    def create_sphere_quads(self, center=(0,0,0), radius=1.0, segments=24, rings=12, name="sphere") -> int:
        """Создаёт сферу и возвращает ID"""
        return self._create_from_factory(
            create_sphere_quads, name,
            center=center, radius=radius, segments=segments, rings=rings
        )

    def create_icosahedron(self, center=(0,0,0), radius=1.0, name="icosahedron") -> int:
        """Создаёт икосаэдр  и возвращает ID"""
        return self._create_from_factory(
            create_icosahedron, name,
            center=center, radius=radius
        )

    def move_point(self, pid: int, delta: np.ndarray) -> None:
        if pid in self.scene.points:
            self.scene.points[pid].position += delta

    def move_point_to(self, pid: int, position: np.ndarray) -> None:
        """Перемещает точку в абсолютную позицию"""
        if pid in self.scene.points:
            self.scene.points[pid].position = position.copy()

    def section_by_plane(self, normal=(0, 0, 1), offset: float = 0.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Вычисляет сечение 3D-модели плоскостью"""
        normal = as_vector3(normal, name="normal")
        norm = float(np.linalg.norm(normal))
        if norm < EPSILON:
            raise ValueError("Section normal cannot be zero vector")
        normal = normal / norm
        offset = float(offset)

        segments = []
        seen_segments = set()

        for face in self.scene.faces.values():
            vertices = [
                self.scene.points[point_id].position
                for point_id in face.vertex_ids
                if point_id in self.scene.points
            ]
            if len(vertices) < 3:
                continue

            face_segments = self._section_face(vertices, normal, offset)
            for start, end in face_segments:
                if np.linalg.norm(end - start) <= EPSILON:
                    continue
                key = self._segment_key(start, end)
                if key in seen_segments:
                    continue
                seen_segments.add(key)
                segments.append((start, end))

        return segments

    def _section_face(self, vertices: List[np.ndarray], normal: np.ndarray,
                      offset: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Вычисляет пересечение плоскости с одной гранью"""
        distances = [float(np.dot(normal, vertex) + offset) for vertex in vertices]
        on_plane = [abs(distance) <= EPSILON for distance in distances]

        if all(on_plane):
            return [
                (vertices[i].copy(), vertices[(i + 1) % len(vertices)].copy())
                for i in range(len(vertices))
            ]

        points = []
        for i, start in enumerate(vertices):
            end = vertices[(i + 1) % len(vertices)]
            d_start = distances[i]
            d_end = distances[(i + 1) % len(vertices)]

            if abs(d_start) <= EPSILON:
                self._append_unique_point(points, start)

            if d_start * d_end < -(EPSILON * EPSILON):
                t = d_start / (d_start - d_end)
                self._append_unique_point(points, start + t * (end - start))
            elif abs(d_end) <= EPSILON:
                self._append_unique_point(points, end)

        if len(points) < 2:
            return []
        if len(points) == 2:
            return [(points[0], points[1])]
        return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]

    def _append_unique_point(self, points: List[np.ndarray], point: np.ndarray) -> None:
        """Добавляет точку в список только если она уникальна (с точностью EPSILON)"""
        for existing in points:
            if np.allclose(existing, point, atol=EPSILON):
                return
        points.append(point.copy())

    def _segment_key(self, start: np.ndarray, end: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Создает уникальный хэшируемый ключ для отрезка без учета направления"""
        start_key = tuple(np.round(start / EPSILON).astype(int))
        end_key = tuple(np.round(end / EPSILON).astype(int))
        return tuple(sorted((start_key, end_key)))

    def clear(self) -> None:
        self.scene = Scene()

    def save(self, path: Path) -> None:
        self.scene.save_json(path)

    def load(self, path: Path) -> None:
        self.scene = Scene.load_json(path)
