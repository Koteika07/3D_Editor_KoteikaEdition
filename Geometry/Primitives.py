"""
Модуль с примитивами для 2D/3D геометрии.

Содержит классы:
- Point: точка в пространстве
- Edge: отрезок между двумя точками
- Plane: плоскость в пространстве

"""

from dataclasses import dataclass, field
from typing import Optional
from __future__ import annotations # преващает аннотацию в строки (лениво)
import numpy as np


class Point:
    """Base class for Point in 3d space"""


    def __init__(self, id: int, position: np.ndarray):
        """Конструктор класса"""
        self.id = id
        self.position = position

        if self.position.shape != (3,):
            raise  ValueError("Position don't have 3 coordinate")


    def __repr__(self) -> str:
        """Вывести объект в виде строки"""
        return f"Point(id={self.id}, position={self.position.tolist()})"


    def __eq__(self, other: object) -> bool:
        """Сравнение обектов =="""
        if not isinstance(other, Point): # проверяет принадлежит ли объект указанному типу (классу)
            return False

        if self.id != self.id:
            return False

        return np.array_equal(self.position, other.position)


    def __hash__(self) -> int:
        """Хеш"""
        return hash((self.id, tuple(self.position)))


    def distance_to(self, other: Point) -> float:
        """Растояние до другой точки"""
        return np.linalg.norm(self.position - other.position)


    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            'id': self.id,
            'position': self.position.tolist()
        }


    @classmethod
    def from_dict(cls, data: dict) -> Point:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id = data['id'],
            position = np.array(data['position'])
        )


@dataclass # декоратор, в нашем случае автоматически создает
# __init__ __repr__ __eq__
class Edge:
    """Base class for Edge in 3d space"""
    id: int
    point_1_id: int
    point_2_id: int


    def __post_init__(self):
        """Валидация после создания (p1 != p2)"""
        if self.point_1_id == self.point_2_id:
            raise ValueError("Edge can't connect a point to itself")


    def get_other_point(self, point_id: int) -> int:
        """Вернуть id другого конца данного отрезка"""
        if point_id == self.point_1_id:
            return self.point_2_id
        elif point_id == self.point_2_id:
            return self.point_1_id
        else:
            raise ValueError(f"Point {point_id} is not on this edge")


    def length(self, points_dict: dict) -> float:
        """Вычислить длину отрезка"""
        p1 = points_dict[self.point_1_id].position
        p2 = points_dict[self.point_2_id].position
        return np.linalg.norm(p1 - p2)


    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            'id': self.id,
            'point_1_id': self.point_1_id,
            'point_2_id': self.point_2_id
        }


    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data['id'],
            point_1_id=data['point_1_id'],
            point_2_id=data['point_2_id']
        )

@dataclass
class Plane:
    """Base class for Plane in 3d space"""
    id: int
    point_1_id: int
    point_2_id: int
    point_3_id: int
    # optional - нужный тип или None
    # field - рассишеренная нстройка dataclasses
    _normal: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _d: Optional[float] = field(default=None, init=False, repr=False)


    def __post_init__(self):
        """Валидация после создания (p1 != p2)"""
        points_set = {self.point_1_id, self.point_2_id, self.point_3_id}
        if len(points_set) < 3:
            raise ValueError("The plane requires 3 different points")


    def compute_normal(self, points_dict: dict) -> np.ndarray:
        """Вычисляет нормаль плоскости"""
        p1 = points_dict[self.point_1_id].position
        p2 = points_dict[self.point_2_id].position
        p3 = points_dict[self.point_3_id].position

        v1 = p2 - p1
        v2 = p3 - p1

        normal = np.cross(v1, v2) # вычисляем нормаль
        norm = np.linalg.norm(normal) # длина вектор

        if norm < 1e-6:
            raise ValueError("Точки коллинеарны, плоскость не определена")

        return normal / norm # нормализируем нормаль


    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            'id': self.id,
            'point_1_id': self.point_1_id,
            'point_2_id': self.point_2_id,
            'point_3_id': self.point_3_id
        }


    @classmethod
    def from_dict(cls, data: dict) -> Plane:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data['id'],
            point_1_id=data['point_1_id'],
            point_2_id=data['point_2_id'],
            point_3_id=data['point_3_id']
        )
