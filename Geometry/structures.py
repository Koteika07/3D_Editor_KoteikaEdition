"""
Модуль с конструкциями для 2D/3D геометрии.

Содержит класс:
- Face: n-мерная face
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from geometry.primitives import Point, Edge, Plane
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class Face:
    """Base class for Face in 3d space"""
    """
    Может быть:
    - Треугольником (3 Edge)
    - Четырехугольником (4 Edge)
    - Многоугольником (n Edge)
    """

    id: int
    edge_ids: List[int]
    plane_id: int

    def __post_init__(self):
        """Валидация после создания"""
        if len(self.edge_ids) < 3:
            raise ValueError(f"Face must have at least 3 edges, have {len(self.edge_ids)}")
        # проверка на дубликаты ребер
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError(f"Duplicate edge IDs in face: {self.edge_ids}")

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
        )
