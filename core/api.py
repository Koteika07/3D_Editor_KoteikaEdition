"""Упрощённое API для работы со сценой"""

from pathlib import Path
from typing import List
import numpy as np

from geometry import Point, Edge, Face, Polyhedron, create_cube, create_pyramid, create_sphere, create_sphere_quads, create_icosahedron
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

    def clear(self) -> None:
        self.scene = Scene()

    def save(self, path: Path) -> None:
        self.scene.save_json(path)

    def load(self, path: Path) -> None:
        self.scene = Scene.load_json(path)
