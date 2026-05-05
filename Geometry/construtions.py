"""
Геометрические построения и конструкторы объектов

Этот модуль отвечает за создание геометрических объектов:
- Примитивов (точки, отрезки, плоскости) по различным правилам
- Составных структур (грани, многогранники)
- Вспомогательных построений (проекции, пересечения)

Принцип: функции принимают ID существующих объектов и API
создают новые объекты через API и возвращают их ID.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np

from geometry.primitives import Point, Edge, Plane
from geometry.structures import Face, Polyhedron