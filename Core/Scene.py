"""
Модуль сцены - контейнер всех данных
Хранит все 3д обекты в данной scene
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from typing import List, Set, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from geometry.primitives import Point, Edge, Plane
from geometry.structures import Face


class ObjectType(Enum):
    """Все типы объектов в сцене"""
    # строковое представление всех типов для удобства сериализации
    POINT = "point"
    EDGE = "edge"
    PLANE = "plane"
    FACE = "face"


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


class Scene:
    """Главное хранилище данных сцены"""

    def __init__(self):
        # хранилища объектов
        self.points: Dict[int, Point] = {}
        self.edges: Dict[int, Edge] = {}
        self.planes: Dict[int, Plane] = {}
        self.faces: Dict[int, Face] = {}

        # счетчики ID (для автоматической генерации)
        self._next_id: Dict[ObjectType, int] = {
            ObjectType.POINT: 0,
            ObjectType.EDGE: 0,
            ObjectType.PLANE: 0,
            ObjectType.FACE: 0
        }

        self.selection = Selection()
        #  заглушка для будущей системы Undo/Redo
        self._history = []

    def _generate_id(self, obj_type: ObjectType) -> int:
        """Сгенерировать новый ID для объекта"""
        # увеличивает счётчик и возвращает новый ID
        self._next_id[obj_type] += 1
        return self._next_id[obj_type]

    # ==================================================================
    # создание объектов
    # ==================================================================

    def add_point(self, position: np.ndarray, custom_id: Optional[int] = None) -> int:
        """Создать точку"""
        # if нету ID то создаём новый или берем существующий
        point_id = custom_id or self._generate_id(ObjectType.POINT)
        self.points[point_id] = Point(point_id, position)
        return point_id

    def add_edge(self, point_1_id: int, point_2_id: int, custom_id: Optional[int] = None) -> int:
        """Добавить отрезок между двумя существующими точками"""
        # проверка существования точек
        if point_1_id not in self.points:
            raise KeyError(f"Point {point_1_id} not found")
        if point_2_id not in self.points:
            raise KeyError(f"Point {point_2_id} not found")
        # if нету ID то создаём новый или берем существующий
        edge_id = custom_id or self._generate_id(ObjectType.EDGE)
        self.edges[edge_id] = Edge(edge_id, point_1_id, point_2_id)
        return edge_id

    def add_plane(self, point_1_id: int, point_2_id: int, point_3_id: int, 
                  custom_id: Optional[int] = None) -> int:
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
        """
        self.points — словарь {id: PointObject}
        .values() — получаем все объекты Point
        p.to_dict() — вызываем метод каждого объекта Point
        на выходе список словарей
        ==================================================
        "points": [
            {"id": 1, "position": [0,0,0]},
            {"id": 2, "position": [1,0,0]}
        ]
        """
        return {
            "points": [p.to_dict() for p in self.points.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "planes": [p.to_dict() for p in self.planes.values()],
            "faces": [f.to_dict() for f in self.faces.values()],
            "next_ids": {k.value: v for k, v in self._next_id.items()}
            # next_ids нужно чтобы при открытии сцены,
            # программа не забыла какой id для нужного типа создать следующим
        }
