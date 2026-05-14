"""
Фабрики геометрических фигур

Создает данные объекты:
* Куб
* Пирамиду
* Сферу
* Икикосаэдр

"""

from typing import List, Tuple
from geometry import EPSILON
import numpy as np
import math


def _normalize_face_orientation(vertices: List[np.ndarray], face: List[int], center: np.ndarray) -> List[int]:
    """Нормализует ориентацию грани (все грани должны быть ориентированы наружу)"""
    if len(face) < 3:
        return face

    p0, p1, p2 = [vertices[i] for i in face[:3]]

    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < EPSILON:
        return face
    normal = normal / norm

    face_center = np.mean([vertices[i] for i in face], axis=0)
    to_center = center - face_center

    # если нормаль смотрит внутрь -> разворачиваем
    if np.dot(normal, to_center) > 0:
        return list(reversed(face))
    return face


def _ensure_consistent_orientation(vertices: List[np.ndarray], faces: List[List[int]], center: np.ndarray) -> List[List[int]]:
    """Обеспечивает согласованность ориентации всех граней"""
    return [_normalize_face_orientation(vertices, face, center) for face in faces]


#=======================| Фабрики объектов |=========================== 
def create_cube(center=(0, 0, 0), size=1.0) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Возвращает (вершины, грани) куба"""
    c = np.array(center)
    h = size / 2

    vertices = [
        c + [-h, -h,  h],  # 0
        c + [ h, -h,  h],  # 1
        c + [ h,  h,  h],  # 2
        c + [-h,  h,  h],  # 3
        c + [-h, -h, -h],  # 4
        c + [ h, -h, -h],  # 5
        c + [ h,  h, -h],  # 6
        c + [-h,  h, -h],  # 7
    ]

    faces = [
        [0, 1, 2, 3],  # передняя
        [5, 4, 7, 6],  # задняя
        [4, 0, 3, 7],  # левая
        [1, 5, 6, 2],  # правая
        [4, 5, 1, 0],  # нижняя
        [3, 2, 6, 7],  # верхняя
    ]

    faces = _ensure_consistent_orientation(vertices, faces, c)
    return vertices, faces


def create_pyramid(center=(0, 0, 0), base_size=1.0, height=1.0) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Возвращает (вершины, грани) пирамиды"""
    c = np.array(center)
    h = base_size / 2
    z_base = -h/2

    vertices = [
        c + [-h, -h, z_base],  # 0
        c + [ h, -h, z_base],  # 1
        c + [ h,  h, z_base],  # 2
        c + [-h,  h, z_base],  # 3
        c + [0, 0, height + z_base],  # 4
    ]

    faces = [
        [0, 1, 2, 3],  # основание
        [0, 1, 4],     # передняя
        [1, 2, 4],     # правая
        [2, 3, 4],     # задняя
        [3, 0, 4],     # левая
    ]
    # центр масс пирамиды (смещён вверх на 1/3 высоты от основания)
    center_obj = c + [0, 0, height/3]
    faces = _ensure_consistent_orientation(vertices, faces, center_obj)
    return vertices, faces


def create_sphere(center=(0, 0, 0), radius=1.0, segments=24, rings=12) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Возвращает (вершины, грани) сферы из треугольников"""
    c = np.array(center)
    vertices, faces = [], []

    # генерация вершин
    for ring in range(rings + 1):
        teta = math.pi * ring / rings
        for seg in range(segments):
            phi = 2 * math.pi * seg / segments
            x = radius * math.sin(teta) * math.cos(phi)
            y = radius * math.sin(teta) * math.sin(phi)
            z = radius * math.cos(teta)
            vertices.append(c + [x, y, z])

    # генерация треугольников
    for ring in range(rings):
        for seg in range(segments):
            i1 = ring * segments + seg
            i2 = (ring + 1) * segments + seg
            i3 = (ring + 1) * segments + ((seg + 1) % segments)
            i4 = ring * segments + ((seg + 1) % segments)

            faces.append([i1, i2, i3])
            faces.append([i1, i3, i4])

    return vertices, faces


def create_sphere_quads(center=(0, 0, 0), radius=1.0, segments=12, rings=6) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Возвращает (вершины, грани) сферы из четырёхугольников"""
    c = np.array(center)
    vertices, faces = [], []

    # генерация вершин
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        for seg in range(segments):
            phi = 2 * math.pi * seg / segments
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            vertices.append(c + [x, y, z])

    # генерация четырёхугольников
    for ring in range(rings):
        for seg in range(segments):
            i1 = ring * segments + seg
            i2 = (ring + 1) * segments + seg
            i3 = (ring + 1) * segments + ((seg + 1) % segments)
            i4 = ring * segments + ((seg + 1) % segments)

            faces.append([i1, i2, i3, i4])

    return vertices, faces


def create_icosahedron(center=(0,0,0), radius=1.0) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Создаёт икосаэдр - 20 граней, работает отлично"""
    c = np.array(center)
    phi = (1 + math.sqrt(5)) / 2.0

    # 12 вершин
    vertices_raw = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]

    # нормализация по радиусу
    vertices = []
    for v in vertices_raw:
        norm = np.linalg.norm(v)
        vertices.append(c + (np.array(v) / norm) * radius)

    # 20 треугольных граней
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    return vertices, faces
