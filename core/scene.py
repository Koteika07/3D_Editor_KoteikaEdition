"""
Модуль сцены - контейнер всех данных
сохраняет и экспортирует в json
"""

from __future__ import annotations  # преващает аннотацию в строки (лениво)
from typing import Dict, Optional
from pathlib import Path
from enum import Enum
import json

from geometry import Edge, Plane, Point, Face, Polyhedron


class ObjectType(Enum):
    """Все типы объектов в сцене"""
    # строковое представление всех типов для удобства сериализации
    POINT = "point"
    EDGE = "edge"
    PLANE = "plane"
    FACE = "face"
    POLYHEDRON = "polyhedron"


class Scene:
    """Главное хранилище данных сцены"""

    def __init__(self):
        # хранилища объектов
        self.points: Dict[int, Point] = {}
        self.edges: Dict[int, Edge] = {}
        self.planes: Dict[int, Plane] = {}
        self.faces: Dict[int, Face] = {}
        self.polyhedra: Dict[int, Polyhedron] = {}
        self._next_id: Dict[ObjectType, int] = {t: 0 for t in ObjectType}

        # счетчики ID (для автоматической генерации)
        self._next_id: Dict[ObjectType, int] = {
            ObjectType.POINT: 0,
            ObjectType.EDGE: 0,
            ObjectType.PLANE: 0,
            ObjectType.FACE: 0,
            ObjectType.POLYHEDRON: 0,
        }

    def _reserve_id(self, obj_type: ObjectType,
                    explicit_id: Optional[int] = None) -> int:
        """Резервирует новый ID для объекта указанного типа"""
        new_id = (
            int(explicit_id)
            if explicit_id is not None
            else (self._next_id[obj_type] + 1)
        )
        self._next_id[obj_type] = max(self._next_id[obj_type], new_id)
        return new_id

    def _get_container(self, obj_type: ObjectType) -> Dict:
        # получем словарь-контейнер который является хранилищем для нашего типа по имени объекта
        if obj_type is ObjectType.POLYHEDRON:
            return self.polyhedra
        return getattr(self, f"{obj_type.value}s")

    def add(self, obj_type: ObjectType, obj_id: int, obj: object) -> None:
        self._get_container(obj_type)[obj_id] = obj

    def remove(self, obj_type: ObjectType, obj_id: int) -> None:
        self._get_container(obj_type).pop(obj_id, None)

    def to_dict(self) -> dict:
        """Сериализация: Object -> словарь для JSON"""
        # .values() — получаем все объекты Point
        # "points": [
        #    {"id": 1, "position": [0,0,0]},
        #    {"id": 2, "position": [1,0,0]}]
        return {
            "points": [p.to_dict() for p in self.points.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "planes": [p.to_dict() for p in self.planes.values()],
            "faces": [f.to_dict() for f in self.faces.values()],
            "polyhedra": [p.to_dict() for p in self.polyhedra.values()],
            "next_ids": {k.value: v for k, v in self._next_id.items()}
            # next_ids нужно чтобы при открытии сцены,
            # программа не забыла какой id для нужного типа создать следующим
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        scene = cls()
        # восстанавливает счётчики ID для каждого типа объектов
        # max защищает от уменьшения счётчика при загрузке
        for key, value in data.get("next_ids", {}).items():
            scene._next_id[ObjectType(key)] = max(
                scene._next_id[ObjectType(key)],
                int(value)
            )
        for p_data in data.get("points", []):
            obj = Point.from_dict(p_data)
            scene.add(ObjectType.POINT, obj.id, obj)
            scene._reserve_id(ObjectType.POINT, obj.id)

        for e_data in data.get("edges", []):
            obj = Edge.from_dict(e_data)
            scene.add(ObjectType.EDGE, obj.id, obj)
            scene._reserve_id(ObjectType.EDGE, obj.id)

        for pl_data in data.get("planes", []):
            obj = Plane.from_dict(pl_data)
            scene.add(ObjectType.PLANE, obj.id, obj)
            scene._reserve_id(ObjectType.PLANE, obj.id)

        for f_data in data.get("faces", []):
            obj = Face.from_dict(f_data)
            scene.add(ObjectType.FACE, obj.id, obj)
            scene._reserve_id(ObjectType.FACE, obj.id)

        for poly_data in data.get("polyhedra", []):
            obj = Polyhedron.from_dict(poly_data)
            scene.add(ObjectType.POLYHEDRON, obj.id, obj)
            scene._reserve_id(ObjectType.POLYHEDRON, obj.id)

        return scene

    def save_json(self, path: Path) -> None:
        path = Path(path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
