"""Тесты для структур: Face, Polyhedron"""

from geometry import Face, Polyhedron
import pytest

class TestFace:
    def test_creation(self):
        f = Face(id=1, vertex_ids=[0, 1, 2], plane_id=10, edge_ids=[100, 101, 102])
        assert f.id == 1
        assert f.vertex_ids == [0, 1, 2]
        assert f.plane_id == 10
        assert f.edge_ids == [100, 101, 102]

    def test_raises_on_less_than_three_vertices(self):
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Face(id=1, vertex_ids=[0, 1], plane_id=10)

    def test_raises_on_duplicate_vertices(self):
        with pytest.raises(ValueError, match="Duplicate vertex"):
            Face(id=1, vertex_ids=[0, 1, 0], plane_id=10)

    def test_get_vertex_ids_closed(self):
        f = Face(id=1, vertex_ids=[0, 1, 2, 3], plane_id=10)
        closed = f.get_vertex_ids(closed=True)
        assert closed == [0, 1, 2, 3, 0]

    def test_contains_point(self):
        f = Face(id=1, vertex_ids=[5, 10, 15], plane_id=10)
        assert f.contains_point(10) is True
        assert f.contains_point(20) is False

    def test_contains_edge(self):
        f = Face(id=1, vertex_ids=[0, 1, 2], plane_id=10, edge_ids=[100, 101, 102])
        assert f.contains_edge(101) is True
        assert f.contains_edge(200) is False

    def test_serialization(self):
        f = Face(id=2, vertex_ids=[0, 1, 2], plane_id=5, edge_ids=[10, 11, 12])
        data = f.to_dict()
        assert data["id"] == 2
        assert data["vertex_ids"] == [0, 1, 2]
        restored = Face.from_dict(data)
        assert restored.id == 2
        assert restored.vertex_ids == [0, 1, 2]
        assert restored.plane_id == 5


class TestPolyhedron:
    def test_creation(self):
        p = Polyhedron(id=1, face_ids=[10, 20, 30, 40], name_object="Cube")
        assert p.id == 1
        assert p.face_ids == [10, 20, 30, 40]
        assert p.name_object == "Cube"

    def test_raises_on_empty_faces(self):
        with pytest.raises(ValueError, match="at least one face"):
            Polyhedron(id=1, face_ids=[])

    def test_raises_on_duplicate_faces(self):
        with pytest.raises(ValueError, match="Duplicate faces"):
            Polyhedron(id=1, face_ids=[10, 20, 10])

    def test_serialization(self):
        p = Polyhedron(id=3, face_ids=[1, 2, 3], name_object="Pyramid")
        data = p.to_dict()
        assert data["id"] == 3
        assert data["face_ids"] == [1, 2, 3]
        assert data["name_object"] == "Pyramid"
        restored = Polyhedron.from_dict(data)
        assert restored.id == 3
        assert restored.face_ids == [1, 2, 3]
        assert restored.name_object == "Pyramid"
