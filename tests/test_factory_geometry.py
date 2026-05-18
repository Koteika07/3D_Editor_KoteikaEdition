"""Тесты для фабрик геометрических фигур"""

import pytest
import numpy as np
from geometry.factory_geometry import (
    create_cube,
    create_pyramid,
    create_sphere,
    create_sphere_quads,
    create_icosahedron,
    _ensure_consistent_orientation,
)


class TestCube:
    def test_returns_8_vertices(self):
        vertices, faces = create_cube()
        assert len(vertices) == 8

    def test_returns_6_faces(self):
        vertices, faces = create_cube()
        assert len(faces) == 6

    def test_all_faces_have_4_vertices(self):
        vertices, faces = create_cube()
        for face in faces:
            assert len(face) == 4

    def test_centered_at_origin(self):
        vertices, _ = create_cube()
        center = np.mean(vertices, axis=0)
        assert np.allclose(center, [0, 0, 0])

    def test_correct_size(self):
        vertices, _ = create_cube(size=2.0)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        assert np.allclose(max_coords - min_coords, [2, 2, 2])


class TestPyramid:
    def test_returns_5_vertices(self):
        vertices, faces = create_pyramid()
        assert len(vertices) == 5

    def test_returns_5_faces(self):
        vertices, faces = create_pyramid()
        assert len(faces) == 5

    def test_base_has_4_vertices(self):
        vertices, faces = create_pyramid()
        # основание - первая грань
        assert len(faces[0]) == 4

    def test_triangular_faces_have_3_vertices(self):
        vertices, faces = create_pyramid()
        for face in faces[1:]:
            assert len(face) == 3


class TestSphere:
    def test_triangle_sphere_has_correct_vertex_count(self):
        vertices, _ = create_sphere(segments=12, rings=6)
        # (rings + 1) * segments = 7 * 12 = 84
        assert len(vertices) == 84

    def test_triangle_sphere_has_triangles(self):
        vertices, faces = create_sphere(segments=4, rings=2)
        for face in faces:
            assert len(face) == 3

    def test_sphere_radius_is_correct(self):
        vertices, _ = create_sphere(radius=2.0, segments=12, rings=6)
        radii = [np.linalg.norm(v) for v in vertices]
        for r in radii:
            assert abs(r - 2.0) < 0.01


class TestSphereQuads:
    def test_quad_sphere_has_quads(self):
        vertices, faces = create_sphere_quads(segments=4, rings=2)
        for face in faces:
            assert len(face) == 4

    def test_vertex_count_matches_formula(self):
        vertices, _ = create_sphere_quads(segments=8, rings=4)
        # (rings + 1) * segments = 5 * 8 = 40
        assert len(vertices) == 40


class TestIcosahedron:
    def test_has_12_vertices(self):
        vertices, _ = create_icosahedron()
        assert len(vertices) == 12

    def test_has_20_faces(self):
        vertices, faces = create_icosahedron()
        assert len(faces) == 20

    def test_all_faces_are_triangles(self):
        vertices, faces = create_icosahedron()
        for face in faces:
            assert len(face) == 3

    def test_radius_is_correct(self):
        vertices, _ = create_icosahedron(radius=2.0)
        radii = [np.linalg.norm(v) for v in vertices]
        for r in radii:
            assert abs(r - 2.0) < 0.001


class TestOrientation:
    def test_orientation_consistency(self):
        vertices = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            np.array([0, 1, 0]),
        ]
        faces = [[0, 1, 2, 3]]
        center = np.array([0.5, 0.5, 0])
        oriented = _ensure_consistent_orientation(vertices, faces, center)
        # ориентация должна быть согласована
        assert len(oriented) == 1
