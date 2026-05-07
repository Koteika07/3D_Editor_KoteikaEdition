"""
Модуль сцены - контейнер всех данных
Хранит все обекты в данной scene
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from typing import Dict, Iterable, Optional
from enum import Enum

import numpy as np

from geometry.primitives import Edge, Plane, Point, as_vector3
from geometry.structures import Face, Polyhedron


class ObjectType(Enum):
    """Все типы объектов в сцене"""
    # строковое представление всех типов для удобства сериализации
    POINT = "point"
    EDGE = "edge"
    PLANE = "plane"
    FACE = "face"
    POLYHEDRON = "polyhedron"


class Scene:
    """Главное хранилище данных сцены"""

    def __init__(self):
        # хранилища объектов
        self.points: Dict[int, Point] = {}
        self.edges: Dict[int, Edge] = {}
        self.planes: Dict[int, Plane] = {}
        self.faces: Dict[int, Face] = {}
        self.polyhedra: Dict[int, Polyhedron] = {}

        # счетчики ID (для автоматической генерации)
        self._next_id: Dict[ObjectType, int] = {
            ObjectType.POINT: 0,
            ObjectType.EDGE: 0,
            ObjectType.PLANE: 0,
            ObjectType.FACE: 0,
            ObjectType.POLYHEDRON: 0,
        }

        # self.selection = Selection()
        # #  заглушка для будущей системы Undo/Redo
        # self._history = []

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

    # ==================================================================
    # сохранение сцены
    # ==================================================================
    def to_dict(self) -> dict:
         """Сериализация: Object -> словарь для JSON"""
        
        # self.points — словарь {id: PointObject}
        # .values() — получаем все объекты Point
        # p.to_dict() — вызываем метод каждого объекта Point
        # на выходе список словарей
        # "points": [
        #    {"id": 1, "position": [0,0,0]},
        #    {"id": 2, "position": [1,0,0]}]WWWWWWWW
        return {
            "points": [point.to_dict() for point in self.points.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
            "planes": [plane.to_dict() for plane in self.planes.values()],
            "faces": [face.to_dict() for face in self.faces.values()],
            "polyhedra": [polyhedron.to_dict() for polyhedron in self.polyhedra.values()],
            "next_ids": {obj_type.value: next_id for obj_type, next_id in self._next_id.items()}
            # next_ids нужно чтобы при открытии сцены,
            # программа не забыла какой id для нужного типа создать следующим
        }
