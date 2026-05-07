"""Пакет geometry: примитивы и составные геометрические объекты"""

from geometry.primitives import EPSILON, Edge, Plane, Point
from geometry.structures import Face, Polyhedron

__all__ = [
    "EPSILON",
    "Point",
    "Edge",
    "Plane",
    "Face",
    "Polyhedron",
]
