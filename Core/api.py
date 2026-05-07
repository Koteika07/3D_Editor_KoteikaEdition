

from typing import Tuple, List, Optional, Set
import numpy as np

from core.scene import Scene

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
