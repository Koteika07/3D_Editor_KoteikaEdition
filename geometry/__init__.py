"""
Пакет geometry: примитивы и составные геометрические объекты
"""

from geometry.primitives import EPSILON, Edge, Plane, Point, as_vector3
from geometry.structures import Face, Polyhedron
from geometry.factory_geometry import (
    create_cube,
    create_pyramid,
    create_sphere,
    create_sphere_quads,
    create_icosahedron
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
    "create_sphere_quads",
    "create_icosahedron",
    "as_vector3",
]