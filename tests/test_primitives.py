"""Тесты для базовых примитивов: Point, Edge, Plane"""

from geometry import EPSILON, Point, Edge, Plane, as_vector3
import numpy as np
import pytest


# assert p.id == 1  # проверка равенства
# assert p1 != p3  # проверка неравенства
# assert isinstance(result, np.ndarray)  # проверка типа
# assert np.allclose(result, [1, 2, 3])  # проверка с плавающей точкой
# pytest.raises - проверка исключений

class TestAsVector3:
    def test_converts_list_to_numpy(self):
        result = as_vector3([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, [1, 2, 3])

    def test_raises_on_wrong_dimension(self):
        with pytest.raises(ValueError, match="doesn't have 3 coordinates"):
            as_vector3([1, 2])
        with pytest.raises(ValueError, match="doesn't have 3 coordinates"):
            as_vector3([1, 2, 3, 4])


class TestPoint:
    def test_creation(self):
        p = Point(1, [1.0, 2.0, 3.0])
        assert p.id == 1
        assert np.allclose(p.position, [1, 2, 3])

    def test_move_delta(self):
        p = Point(1, [0, 0, 0])
        p.move([1, 2, 3])
        assert np.allclose(p.position, [1, 2, 3])

    def test_distance_to(self):
        p1 = Point(1, [0, 0, 0])
        p2 = Point(2, [3, 4, 0])
        assert abs(p1.distance_to(p2) - 5.0) < EPSILON

    def test_eq_with_tolerance(self):
        p1 = Point(1, [1.0, 2.0, 3.0])
        p2 = Point(1, [1.0000001, 2.0, 3.0])
        p3 = Point(2, [1, 2, 3])
        assert p1 == p2
        assert p1 != p3

    def test_serialization(self):
        p = Point(5, [1.5, 2.5, 3.5])
        data = p.to_dict()
        assert data == {"id": 5, "position": [1.5, 2.5, 3.5]}
        restored = Point.from_dict(data)
        assert restored.id == 5
        assert np.allclose(restored.position, [1.5, 2.5, 3.5])


class TestEdge:
    def test_creation(self):
        e = Edge(1, 0, 1)
        assert e.id == 1
        assert e.point_1_id == 0
        assert e.point_2_id == 1

    def test_raises_on_same_points(self):
        with pytest.raises(ValueError, match="can't connect a point to itself"):
            Edge(1, 5, 5)

    def test_key_is_order_independent(self):
        e1 = Edge(1, 5, 10)
        e2 = Edge(2, 10, 5)
        assert e1.key() == e2.key()
        assert e1.key() == frozenset({5, 10})

    def test_get_other_point(self):
        e = Edge(1, 5, 10)
        assert e.get_other_point(5) == 10
        assert e.get_other_point(10) == 5
        with pytest.raises(ValueError, match="is not on this edge"):
            e.get_other_point(7)

    def test_contains_point(self):
        e = Edge(1, 5, 10)
        assert e.contains_point(5) is True
        assert e.contains_point(10) is True
        assert e.contains_point(7) is False

    def test_serialization(self):
        e = Edge(3, 1, 2)
        data = e.to_dict()
        assert data == {"id": 3, "point_1_id": 1, "point_2_id": 2}
        restored = Edge.from_dict(data)
        assert restored.id == 3
        assert restored.point_1_id == 1
        assert restored.point_2_id == 2


class TestPlane:
    def test_creation_by_normal(self):
        p = Plane(1, point_1_id=0, normal=[0, 0, 1])
        assert p.normal is not None
        assert np.allclose(p.normal, [0, 0, 1])

    def test_creation_without_normal_needs_three_points(self):
        with pytest.raises(ValueError, match="нужны 3 опорные точки"):
            Plane(1, point_1_id=0, point_2_id=1)

    def test_raises_on_zero_normal(self):
        with pytest.raises(ValueError, match="cannot be zero vector"):
            Plane(1, point_1_id=0, normal=[0, 0, 0])

    def test_compute_normal_from_points(self):
        points = {
            1: Point(1, [0, 0, 0]),
            2: Point(2, [1, 0, 0]),
            3: Point(3, [0, 1, 0]),
        }
        p = Plane(1, point_1_id=1, point_2_id=2, point_3_id=3)
        normal = p.compute_normal(points)
        # точки (0,0,0), (1,0,0), (0,1,0) -> нормаль (0,0,1)
        assert np.allclose(normal, [0, 0, 1])

    def test_plane_equation(self):
        points = {
            1: Point(1, [0, 0, 0]),
            2: Point(2, [1, 0, 0]),
            3: Point(3, [0, 1, 0]),
        }
        p = Plane(1, point_1_id=1, point_2_id=2, point_3_id=3)
        normal, d = p.get_plane_equation(points)
        assert np.allclose(normal, [0, 0, 1])
        assert abs(d) < EPSILON

    def test_signed_distance(self):
        points = {
            1: Point(1, [0, 0, 0]),
            2: Point(2, [1, 0, 0]),
            3: Point(3, [0, 1, 0]),
        }
        p = Plane(1, point_1_id=1, point_2_id=2, point_3_id=3)
        dist = p.signed_distance([0, 0, 5], points)
        assert abs(dist - 5) < EPSILON

    def test_project_point(self):
        points = {
            1: Point(1, [0, 0, 0]),
            2: Point(2, [1, 0, 0]),
            3: Point(3, [0, 1, 0]),
        }
        p = Plane(1, point_1_id=1, point_2_id=2, point_3_id=3)
        projected = p.project_point([10, 20, 5], points)
        assert abs(projected[2]) < EPSILON

    def test_serialization(self):
        p = Plane(7, point_1_id=1, normal=[1, 0, 0])
        data = p.to_dict()
        assert data["id"] == 7
        assert data["normal"] == [1, 0, 0]
        restored = Plane.from_dict(data)
        assert restored.id == 7
        assert np.allclose(restored.normal, [1, 0, 0])
