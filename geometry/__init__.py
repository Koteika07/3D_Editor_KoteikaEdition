"""
Пакет geometry: примитивы и составные геометрические объекты
"""

from geometry.primitives import EPSILON, Edge, Plane, Point, as_vector3
from geometry.structures import Face, Polyhedron
from geometry.factory_geometry import (
    create_cube,
    create_pyramid,
    create_sphere,
    create_dodecahedron,
    build_topology
)

__all__ = [
    "EPSILON",
    "Point",
    "Edge",
    "Plane",
    "Face",
    "Polyhedron",
    "create_cube",
    "create_pyramid",
    "create_sphere",
    "create_dodecahedron",
    "build_topology",
    "as_vector3",
]
