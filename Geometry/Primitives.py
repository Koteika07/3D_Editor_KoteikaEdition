"""
Модуль с базовыми примитивами для 2D/3D геометрии.

Содержит классы:
- Point: точка в пространстве
- Edge: отрезок между двумя точками
- Plane: плоскость в пространстве
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
import numpy as np

EPSILON = 1e-6


def as_vector3(value: Sequence[float] | np.ndarray, *, name: str = "vector") -> np.ndarray:
    """Преобразовать входные данные в 3D-вектор numpy(float)"""
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} doesn't have 3 coordinates")
    return vector.copy()


class Point:
    """Base class for Point in 3d space"""

    def __init__(self, id: int, position: np.ndarray):
        """Конструктор класса"""
        self.id = id
        self.position = as_vector3(position, name="position")

    def __repr__(self) -> str:
        """Вывести объект в виде строки"""
        return f"Point(id={self.id}, position={self.position.tolist()})"

    def __eq__(self, other: object) -> bool:
        """Сравнение объектов =="""
        if not isinstance(other, Point):
            return False
        # np.allclose - сравнение с учетом погрешности
        return self.id == other.id and np.allclose(self.position, other.position, atol=EPSILON)

    def __hash__(self) -> int:
        """Хеш точки"""
        # точки, попадающие в одну ячейку(размер = EPSILON), получают одинаковый хеш
        # Точка A: (2.3, 3.7) => округление => (2, 4)
        # Точка C: (2.4, 3.8) => округление => (2, 4)
        rounded = tuple(np.round(self.position / EPSILON).astype(int))
        return hash((self.id, rounded))

    def move(self, delta: np.ndarray) -> None:
        """Переместить точку на вектор delta"""
        self.position = self.position + as_vector3(delta, name="delta")

    def distance_to(self, other: Point) -> float:
        """Расстояние до другой точки"""
        return np.linalg.norm(self.position - other.position)

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "position": self.position.tolist()
        }

    @classmethod
    def from_dict(cls, data: dict) -> Point:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data["id"],
            position=np.array(data["position"])
        )


@dataclass  # декоратор, в нашем случае автоматически создает
# __init__ __repr__ __eq__
# frozen = True => объект неизменяемый (автоматический __hash__)
class Edge(frozenset=True):
    """Base class for Edge in 3d space"""

    id: int
    point_1_id: int
    point_2_id: int

    def __post_init__(self):
        """Валидация после создания (p1 != p2)"""
        if self.point_1_id == self.point_2_id:
            raise ValueError("Edge can't connect a point to itself")

    def key(self) -> frozenset[int]:
        """Вернуть ключ ребра без учета порядка точек."""
        # игнорирует порядок, т.е
        # edge1 = frozenset((5, 10)) => {10, 5}
        # edge2 = frozenset((10, 5)) => {10, 5}
        return frozenset((self.point_1_id, self.point_2_id))

    def get_other_point(self, point_id: int) -> int:
        """Вернуть ID другого конца данного отрезка"""
        if point_id == self.point_1_id:
            return self.point_2_id
        elif point_id == self.point_2_id:
            return self.point_1_id
        else:
            raise ValueError(f"Point {point_id} is not on this edge")

    def contains_point(self, point_id: int) -> bool:
        """Проверяет, принадлежит ли точка отрезку"""
        return point_id == self.point_1_id or point_id == self.point_2_id

    def length(self, points_dict: dict) -> float:
        """Вычислить длину отрезка"""
        p1 = points_dict[self.point_1_id].position
        p2 = points_dict[self.point_2_id].position
        return float(np.linalg.norm(p1 - p2))

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "point_1_id": self.point_1_id,
            "point_2_id": self.point_2_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data["id"],
            point_1_id=data["point_1_id"],
            point_2_id=data["point_2_id"]
        )


@dataclass
class Plane:
    """Base class for Plane in 3d space"""

    """
    Плоскость может быть задана:
        - тремя точками
        - одной точкой и normal-ью, if плоскость параллельна другой
    """

    id: int
    point_1_id: int
    point_2_id: Optional[int] = None
    point_3_id: Optional[int] = None
    # optional - нужный тип или None
    # field - рассишеренная нстройка dataclasses
    _normal: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        """Валидация после создания"""
        self.point_1_id = int(self.point_1_id)
        if self.point_2_id is not None:
            self.point_2_id = int(self.point_2_id)
        if self.point_3_id is not None:
            self.point_3_id = int(self.point_3_id)

        point_ids = self.get_point_ids()
        if len(set(point_ids)) != len(point_ids):
            raise ValueError(f"Repeating points in the plane: {point_ids}")
        if self.normal is None:
            if self.point_2_id is None or self.point_3_id is None:
                raise ValueError("Плоскости без нормали нужны 3 опорные точки")
        else:
            normal = as_vector3(self.normal, name="normal")
            norm = float(np.linalg.norm(normal))
            if norm < EPSILON:
                raise ValueError("Plane normal cannot be zero vector")
            self.normal = normal / norm

    def get_point_ids(self) -> list[int]:
        """Вернуть ID опорных точек плоскости"""
        point_ids = [self.point_1_id]
        if self.point_2_id is not None:
            point_ids.append(self.point_2_id)
        if self.point_3_id is not None:
            point_ids.append(self.point_3_id)
        return point_ids

    def compute_normal(self, points: dict) -> np.ndarray:
        """Вычисляет нормаль плоскости"""
        if self.normal is not None:
            return self.normal.copy()

        if self.point_2_id is None or self.point_3_id is None:
            raise ValueError("3 reference points are needed to calculate the normal")

        p1 = points[self.point_1_id].position
        p2 = points[self.point_2_id].position
        p3 = points[self.point_3_id].position
        normal = np.cross(p2 - p1, p3 - p1)  # вычисляем нормаль
        norm = float(np.linalg.norm(normal))  # длина вектор

        if norm < EPSILON:
            raise ValueError("The points are collinear, the plane is not defined")
        return normal / norm  # нормализируем нормаль

    def get_plane_equation(self, points: Dict[int, Point]) -> tuple[np.ndarray, float]:
        """Возвращает уравнение плоскости: normal*x + d = 0"""
        normal = self.compute_normal(points)
        anchor = points[self.point_1_id].position
        d = -float(np.dot(normal, anchor))
        return normal, d

    def get_normal(self, points: Dict[int, Point]) -> np.ndarray:
        normal, _ = self.get_equation(points)
        return normal

    def get_d(self, points: Dict[int, Point]) -> float:
        _, d = self.get_equation(points)
        return d

    def signed_distance(self, point: np.ndarray, points: Dict[int, Point]) -> float:
        """Вычислияет знаковое расстояние от точки до плоскости"""
        normal, d = self.get_equation(points)
        return float(np.dot(normal, as_vector3(point, name="point")) + d)

    def is_point_on_plane(self, point: np.ndarray, points: Dict[int, Point]) -> bool:
        """Проверяет, лежит ли точка в плоскости"""
        return abs(self.signed_distance(point, points)) <= EPSILON

    def is_parallel_to(self, other: 'Plane', points: Dict[int, Point]) -> bool:
        """Проверяет, параллельны ли плоскости """
        n1 = self.get_normal(points)
        n2 = other.get_normal(points)
        # np.cross(n1, n2) - векторное произведение
        # np.linalg.norm(v) - длина вектора
        return np.linalg.norm(np.cross(n1, n2)) <= EPSILON

    def project_point(self, point: np.ndarray, points: Dict[int, Point]) -> np.ndarray:
        """Проецирует точку на плоскость"""
        normal, d = self.get_plane_equation(points)
        vector = as_vector3(point, name="point")
        distance = float(np.dot(normal, vector) + d)
        return vector - distance * normal

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "point_1_id": self.point_1_id,
            "point_2_id": self.point_2_id,
            "point_3_id": self.point_3_id,
            "normal": None if self.normal is None else self.normal.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Plane:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=int(data["id"]),
            point_1_id=int(data["point_1_id"]),
            point_2_id=int(data["point_2_id"]) if data.get("point_2_id") is not None else None,
            point_3_id=int(data["point_3_id"]) if data.get("point_3_id") is not None else None,
            normal=None if data.get("normal") is None else np.array(data["normal"], dtype=float)
        )
