import numpy as np
from dataclasses import dataclass


class Point:
    """Base class for Point in 3d space"""


    def __init__(self, id: int, position: np.ndarray):
        """
        Конструктор класса 
        """
        self.id = id
        self.position = position

        if self.position.shape != (3,):
            raise  ValueError("Position don't have 3 coordinate")


    def __repr__(self) -> str:
        """
        Вывести объект в виде строки
        """
        return f"Point(id={self.id}, position={self.position.tolist()})"


    def __eq__(self, other: object) -> bool:
        """
        Сравнение обектов ==
        """
        if not isinstance(other, Point): # проверяет принадлежит ли объект указанному типу (классу)
            return False

        if self.id != self.id:
            return False

        return np.array_equal(self.position, other.position)


    def __hash__(self) -> int:
        """
        Хеш
        """
        return hash((self.id, tuple(self.position)))


    def distance_to(self, other: 'Point') -> float:
        """
        Растояние до другой точки
        """
        return np.linalg.norm(self.position - other.position)


    def to_dict(self) -> dict:
        """
        Объект -> словарь для JSON
        """
        return {
            'id': self.id,
            'position': self.position.tolist()
        }


class Egde:
    """Base class for Edge in 3d space"""


class Plane:
    """Base class for Plane in 3d space"""
