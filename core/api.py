"""API для GUI-команд и создание геометрии"""

from __future__ import annotations
from typing import Dict, Optional, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.scene import Scene, ObjectType
from geometry import constructions
from geometry.primitives import Point, Edge, Plane, as_vector3
from geometry.structures import Face, Polyhedron


@dataclass
class Selection:
    """Управление выделенными объектами"""
    # set хранит ID выделенных объектов каждого тип
    # создаёт новый пустой set для каждого экземпляра
    points: Set[int] = field(default_factory=set)
    edges: Set[int] = field(default_factory=set)
    planes: Set[int] = field(default_factory=set)
    faces: Set[int] = field(default_factory=set)

    def clear(self):
        """Очищяет выделение"""
        self.points.clear()
        self.edges.clear()
        self.planes.clear()
        self.faces.clear()

    def is_empty(self) -> bool:
        """Проверяет, есть ли выделенные объекты"""
        # в python пустой set в логическом контексте даёт False
        return not (self.points or self.edges or self.planes or self.faces)

    def get_all_ids(self) -> Dict[ObjectType, Set[int]]:
        """Возвращает все выделенные ID по типам"""
        return {
            ObjectType.POINT: self.points,
            ObjectType.EDGE: self.edges,
            ObjectType.PLANE: self.planes,
            ObjectType.FACE: self.faces
        }


    def _generate_id(self, obj_type: ObjectType) -> int:
        """Сгенерировать новый ID для объекта"""
        # увеличивает счётчик и возвращает новый ID
        self._next_id[obj_type] += 1
        return self._next_id[obj_type]

    def _remember_id(self, obj_type: ObjectType, object_id: int) -> None:
        """Обновляет счётчик ID, если переданный ID больше текущего"""
        self._next_id[obj_type] = max(self._next_id[obj_type], int(object_id))

    def _id(self, obj_type: ObjectType, custom_id: Optional[int]) -> int:
        """Возвращает ID: новый сгенерированный или переданный. Обновляет счётчик"""
        object_id = self._generate_id(obj_type) if custom_id is None else int(custom_id)
        self._remember_id(obj_type, object_id)
        return object_id

    def _require_points(self, point_ids: Iterable[int]) -> None:
        """Проверяет, что все точки с указанными ID существуют в сцене"""
        for point_id in point_ids:
            if point_id not in self.points:
                raise KeyError(f"Point {point_id} not found")

    def _require_edges(self, edge_ids: Iterable[int]) -> None:
        """Проверяет, что все рёбра с указанными ID существуют в сцене"""
        for edge_id in edge_ids:
            if edge_id not in self.edges:
                raise KeyError(f"Edge {edge_id} not found")

    def _require_faces(self, face_ids: Iterable[int]) -> None:
        """Проверяет, что все грани с указанными ID существуют в сцене"""
        for face_id in face_ids:
            if face_id not in self.faces:
                raise KeyError(f"Face {face_id} not found")

    # ==================================================================
    # создание объектов
    # ==================================================================

    def add_point(self, position: np.ndarray, custom_id: Optional[int] = None) -> int:
        """Создать точку"""
        point_id = self._id(ObjectType.POINT, custom_id)
        if point_id in self.points:
            raise ValueError(f"Point {point_id} already exist")
        self.points[point_id] = Point(point_id, as_vector3(position, name="position"))
        return point_id

    def add_edge(self, point_1_id: int, point_2_id: int, custom_id: Optional[int] = None, *, reuse: bool = True) -> int:
        """Создать отрезок между двумя существующими точками"""
        # reuse — фла управляющий повторным использованием уже существующего ребра
        # проверка существования точек
        self._require_points((point_1_id, point_2_id))
        if reuse and custom_id is None:
            existing_id = self.find_edge_between(point_1_id, point_2_id)
            if existing_id is not None:
                return existing_id
        edge_id = self._id(ObjectType.EDGE, custom_id)
        if edge_id in self.edges:
            raise ValueError(f"Edge {edge_id} already exist")
        self.edges[edge_id] = Edge(edge_id, int(point_1_id), int(point_2_id))
        return edge_id

    def add_plane(self, point_ids: Sequence[int], normal: Optional[np.ndarray] = None, custom_id: Optional[int] = None) -> int:
        """Добавить плоскость по трём точкам"""
        # проверяем существование точек
        for pid in [point_1_id, point_2_id, point_3_id]:
            if pid not in self.points:
                raise KeyError(f"Point {pid} not found")

        # if нету ID то создаём новый или берем существующий
        plane_id = custom_id or self._generate_id(ObjectType.PLANE)
        self.planes[plane_id] = Plane(plane_id, point_1_id, point_2_id, point_3_id)
        return plane_id

    def add_face(self, edge_ids: List[int], plane_id: int, custom_id: Optional[int] = None) -> int:
        """Добавить грань (n-угольник)"""
        # проверяем существование ребер
        for eid in edge_ids:
            if eid not in self.edges:
                raise KeyError(f"Edge {eid} not found")

        face_id = custom_id or self._generate_id(ObjectType.FACE)
        self.faces[face_id] = Face(face_id, edge_ids, plane_id)
        return face_id

    # ==================================================================
    # удаление объектов
    # ==================================================================
    def delete_selected(self):
        """Удалить все выделенные объекты"""
        # удаляем в порядке зависимости: точки → отрезки → плоскости → грани
        for point_id in list(self.selection.points):
            self.delete_point(point_id)