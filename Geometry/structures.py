"""
Модуль с конструкциями для 2D/3D геометрии

Содержит классы:
- Face: n-мерная face
- Polyhedron: n-мерные гранники (состоят из face)
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from geometry.primitives import Point


@dataclass
class Face:
    """Base class for Face in 3d space

    Может быть:
    - Треугольником (3 Edge)
    - Четырехугольником (4 Edge)
    - Многоугольником (n Edge)
    
    vertex_ids хранятся в порядке обхода
    edge_ids нужны для выделения и отрисовки ребер как отдельных объектов
    """

    id: int
    vertex_ids: List[int]
    plane_id: int
    # default_factory - механизм создания значений по умолчанию для каждого экземпляра dataclass
    edge_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Валидация после создания"""
        self.vertex_ids = [int(vertex_id) for vertex_id in self.vertex_ids]
        self.edge_ids = [int(edge_id) for edge_id in self.edge_ids]
        if len(self.vertex_ids) < 3:
            raise ValueError(f"Face must have at least 3 vertices, now {len(self.vertex_ids)}")
        if len(set(self.vertex_ids)) != len(self.vertex_ids):
            raise ValueError(f"Duplicate vertex in face: {self.vertex_ids}")
        if self.edge_ids and len(self.edge_ids) != len(self.vertex_ids):
            raise ValueError("Face has count of edge = vertex")

    def get_vertex_ids(self, *, closed: bool = False) -> List[int]:
        """Возвращает ID вершин в порядке обхода"""
        if closed:
            return [*self.vertex_ids, self.vertex_ids[0]]
        return list(self.vertex_ids)

    def get_vertex_positions(self, points: Dict[int, Point]) -> List[np.ndarray]:
        """Возвращает координаты вершин грани"""
        return [points[point_id].position.copy() for point_id in self.vertex_ids]

    def contains_point(self, point_id: int) -> bool:
        """Проверить, входит ли точка в грань"""
        return point_id in self.vertex_ids

    def contains_edge(self, edge_id: int) -> bool:
        """Проверяет, входит ли ребро в грань"""
        return edge_id in self.edge_ids

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "vertex_ids": list(self.vertex_ids),
            "edge_ids": list(self.edge_ids),
            "plane_id": self.plane_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> Face:
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=int(data["id"]),
            vertex_ids=list(data.get("vertex_ids", [])),
            edge_ids=list(data.get("edge_ids", [])),
            plane_id=int(data["plane_id"]),
        )


@dataclass
class Polyhedron:
    """Многогранник, задаётся списком граней"""

    id: int
    face_ids: List[int]
    name_object: str = ""

    def __post_init__(self) -> None:
        """Валидация после создания"""
        self.face_ids = [int(face_id) for face_id in self.face_ids]
        if not self.face_ids:
            raise ValueError("Polyhedron must have at least one face")
        if len(set(self.face_ids)) != len(self.face_ids):
            raise ValueError(f"Duplicate faces in polyhedron: {self.face_ids}")

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        return {
            "id": self.id,
            "face_ids": list(self.face_ids),
            "name_object": self.name_object,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Polyhedron":
        """Десериализация: JSON (dict) -> Object"""
        return cls(
            id=int(data["id"]),
            face_ids=list(data["face_ids"]),
            name_object=data.get("name_object", ""),
        )
