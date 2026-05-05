"""
Модуль с примитивами для 2D/3D геометрии.

Содержит классы:
- Point: точка в пространстве
- Edge: отрезок между двумя точками
- Plane: плоскость в пространстве
<<<<<<< HEAD
=======
- Face: n-мерная face
- Polyhedro: n-гранники
>>>>>>> b2032216cf0da4666052828fcc0537f5ab3b3a44
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

EPSILON = 1e-6


class Point:
    """Base class for Point in 3d space"""

    def __init__(self, id: int, position: np.ndarray):
        """Конструктор класса"""
        self.id = id
        self.position = position

        if self.position.shape != (3,):
            raise ValueError("Position doesn't have 3 coordinates")

    def __repr__(self) -> str:
        """Вывести объект в виде строки"""
        return f"Point(id={self.id}, position={self.position.tolist()})"

    def __eq__(self, other: object) -> bool:
        """Сравнение объектов =="""
        # проверяет принадлежит ли объект указанному типу (классу)
        if not isinstance(other, Point):
            return False

        if self.id != other.id:
            return False

        return np.array_equal(self.position, other.position)

    def __hash__(self) -> int:
        """Хеш"""
        return hash((self.id, tuple(self.position)))

    def distance_to(self, other: Point) -> float:
        """Расстояние до другой точки"""
        return np.linalg.norm(self.position - other.position)

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {"id": self.id, "position": self.position.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> Point:
        """Десериализация: JSON (dict) -> Object"""
        return cls(id=data["id"], position=np.array(data["position"]))


@dataclass  # декоратор, в нашем случае автоматически создает
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
        """Вернуть ID другого конца данного отрезка"""
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

    def contains_point(self, point_id: int) -> bool:
        """Проверяет, принадлежит ли точка отрезку"""
        return point_id == self.point_1_id or point_id == self.point_2_id

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

    id: int
    point_1_id: int
    point_2_id: int
    point_3_id: int
    # optional - нужный тип или None
    # field - рассишеренная нстройка dataclasses
    _normal: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _d: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Валидация после создания"""
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

        normal = np.cross(v1, v2)  # вычисляем нормаль
        norm = np.linalg.norm(normal)  # длина вектор

        if norm < EPSILON:
            raise ValueError("Точки коллинеарны, плоскость не определена")

        return normal / norm  # нормализируем нормаль

    def get_equation(self, points_dict: Dict[int, Point]) -> Tuple[np.ndarray, float]:
        """
        Возвращает уравнение плоскости в виде (нормаль, d).
        Уравнение: n·x + d = 0
        """
        if self._normal is None:
            self._normal = self.compute_normal(points_dict)
            p1 = points_dict[self.point1_id].position  # одна точка из плоскости
            self._d = -np.dot(self._normal, p1)  # считаем коэффициент d

        return self._normal, self._d

    def signed_distance(self, point: np.ndarray, points_dict: Dict[int, Point]) -> float:
        """
        Вычисляет знаковое расстояние от точки до плоскости
        """
        normal, d = self.get_plane_equation(points_dict)
        return np.dot(normal, point) + d

    def is_point_on_plane(self, point: np.ndarray, points_dict: Dict[int, Point]) -> bool:
        """
        Проверяет, лежит ли точка в плоскости.
        """
        return abs(self.signed_distance(point, points_dict)) < EPSILON

    def is_parallel_to(self, other: 'Plane', points_dict: Dict[int, Point]) -> bool:
        """Проверяет, параллельны ли плоскости """
        n1 = self.get_normal(points_dict)
        n2 = other.get_normal(points_dict)
        # np.cross(n1, n2) - векторное произведение
        # np.linalg.norm(v) - длина вектора
        return np.linalg.norm(np.cross(n1, n2)) < EPSILON

    def project_point(self, point: np.ndarray, points_dict: Dict[int, Point]) -> np.ndarray:
        """Проецирует точку на плоскость"""
        normal, d = self.get_plane_equation(points_dict)
        distance = np.dot(normal, point) + d
        return point - distance * normal

    def get_normal(self, points_dict: Dict[int, Point]) -> np.ndarray:
        normal, _ = self.get_plane_equation(points_dict)
        return normal

    def get_d(self, points_dict: Dict[int, Point]) -> float:
        _, d = self.get_plane_equation(points_dict)
        return d

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "point_1_id": self.point_1_id,
            "point_2_id": self.point_2_id,
            "point_3_id": self.point_3_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> Plane:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data["id"],
            point_1_id=data["point_1_id"],
            point_2_id=data["point_2_id"],
            point_3_id=data["point_3_id"]
<<<<<<< HEAD
=======
        )


@dataclass
class Face:
    """Base class for Face in 3d space"""

    id: int
    edge_ids: List[int]
    plane_id: int

    def __post_init__(self):
        """Валидация после создания"""
        if len(self.edge_ids) < 3:
            raise ValueError(f"Face must have at least 3 edges, have {len(self.edge_ids)}")

    def get_vertex_ids(self, edges_dict: Dict[int, Edge]) -> List[int]:
        """Возвращает ID вершин грани в порядке обхода"""
        if not self.edge_ids:
            return []

        # множество для отслеживания уже использованных ребер
        used_edges = set()
        # список для хранения вершин в порядке обход
        vertices = []
        # начальное ребро
        first_edge = edges_dict[self.edge_ids[0]]
        current_vertex = first_edge.point1_id
        vertices.append(current_vertex)
        used_edges.add(self.edge_ids[0])

        # пока не использовали все ребра
        while len(used_edges) < len(self.edge_ids):
            # флаг на следующее ребро
            found = False
            for edge_id in self.edge_ids:
                if edge_id in used_edges:
                    continue
                edge = edges_dict[edge_id]
                # если нашли получаем ID другой точки ребра
                if edge.contains_point(current_vertex):
                    new_vertex = edge.get_other_point(current_vertex)

                    vertices.append(new_vertex)
                    current_vertex = new_vertex
                    used_edges.add(edge_id)
                    found = True
                    break
            if not found:
                raise ValueError(f"Edge not found for vertex {current_vertex}")
        # востновавливаем замыкание
        if vertices[0] != vertices[-1]:
            vertices.append(vertices[0])

        return vertices

    def get_vertex_positions(self, points_dict: Dict[int, Point],
                             edges_dict: Dict[int, Edge]) -> List[np.ndarray]:
        """
        Возвращает координаты вершин грани в порядке обхода.
        """
        vertex_ids = self.get_vertex_ids(edges_dict)
        positions = []

        for v in vertex_ids:
            if v in points_dict:
                point = points_dict[v]
                positions.append(point.position)

        return positions

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            'id': self.id,
            'edge_ids': self.edge_ids,
            'plane_id': self.plane_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> Face:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=data['id'],
            edge_ids=data['edge_ids'],
            plane_id=data['plane_id']
>>>>>>> b2032216cf0da4666052828fcc0537f5ab3b3a44
        )
