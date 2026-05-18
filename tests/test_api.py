"""Тесты для GeometryAPI (сечения, операции с объектами)"""

from core import GeometryAPI
from pathlib import Path
import numpy as np
import tempfile
import pytest


class TestGeometryAPI:
    def setup_method(self):
        self.api = GeometryAPI()

    def test_add_point(self):
        pid = self.api.add_point([1, 2, 3])
        assert pid in self.api.scene.points
        assert np.allclose(self.api.scene.points[pid].position, [1, 2, 3])

    def test_add_edge(self):
        p1 = self.api.add_point([0, 0, 0])
        p2 = self.api.add_point([1, 0, 0])
        eid = self.api.add_edge(p1, p2)
        assert eid in self.api.scene.edges
        assert self.api.scene.edges[eid].point_1_id == p1
        assert self.api.scene.edges[eid].point_2_id == p2

    def test_move_point(self):
        pid = self.api.add_point([0, 0, 0])
        self.api.move_point(pid, [1, 2, 3])
        assert np.allclose(self.api.scene.points[pid].position, [1, 2, 3])

    def test_move_point_to(self):
        pid = self.api.add_point([0, 0, 0])
        self.api.move_point_to(pid, [5, 5, 5])
        assert np.allclose(self.api.scene.points[pid].position, [5, 5, 5])

    def test_create_cube(self):
        pid = self.api.create_cube(center=(0, 0, 0), size=2.0)
        assert pid in self.api.scene.polyhedra
        assert len(self.api.scene.points) == 8
        assert len(self.api.scene.faces) == 6

    def test_clear_scene(self):
        self.api.create_cube()
        assert len(self.api.scene.points) > 0
        self.api.clear()
        assert len(self.api.scene.points) == 0
        assert len(self.api.scene.edges) == 0
        assert len(self.api.scene.faces) == 0
        assert len(self.api.scene.polyhedra) == 0


class TestSectionByPlane:
    def setup_method(self):
        self.api = GeometryAPI()
        # создаём простой куб от -1 до 1
        self.api.create_cube(center=(0, 0, 0), size=2.0)

    def test_section_through_center_xy_plane(self):
        # сечение плоскостью Z=0
        segments = self.api.section_by_plane(normal=(0, 0, 1), offset=0)
        # сечение квадрата даёт 4 отрезка (границы прямоугольника)
        assert len(segments) == 4

    def test_section_above_center_gives_empty(self):
        segments = self.api.section_by_plane(normal=(0, 0, 1), offset=2)
        assert len(segments) == 0

    def test_section_below_center_gives_empty(self):
        segments = self.api.section_by_plane(normal=(0, 0, 1), offset=-2)
        assert len(segments) == 0

    def test_section_yz_plane(self):
        segments = self.api.section_by_plane(normal=(1, 0, 0), offset=0)
        assert len(segments) == 4

    def test_raises_on_zero_normal(self):
        with pytest.raises(ValueError, match="cannot be zero vector"):
            self.api.section_by_plane(normal=(0, 0, 0), offset=0)

    def test_section_with_slanted_plane(self):
        segments = self.api.section_by_plane(normal=(1, 1, 0), offset=0)
        # должно быть пересечение по диагонали
        assert len(segments) >= 2


class TestSerialization:
    def setup_method(self):
        self.api = GeometryAPI()
        self.api.create_cube(name="TestCube")
        self.api.create_sphere(name="TestSphere")

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            self.api.save(path)
            new_api = GeometryAPI()
            new_api.load(path)

            assert len(new_api.scene.points) == len(self.api.scene.points)
            assert len(new_api.scene.faces) == len(self.api.scene.faces)
            assert len(new_api.scene.polyhedra) == len(self.api.scene.polyhedra)
        finally:
            path.unlink(missing_ok=True)

    def test_preserves_point_positions(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            original_positions = {
                pid: p.position.copy()
                for pid, p in self.api.scene.points.items()
            }

            self.api.save(path)
            new_api = GeometryAPI()
            new_api.load(path)

            for pid, pos in original_positions.items():
                assert pid in new_api.scene.points
                assert np.allclose(new_api.scene.points[pid].position, pos)
        finally:
            path.unlink(missing_ok=True)


class TestFaceCreation:
    def setup_method(self):
        self.api = GeometryAPI()

    def test_add_face_with_three_points(self):
        p1 = self.api.add_point([0, 0, 0])
        p2 = self.api.add_point([1, 0, 0])
        p3 = self.api.add_point([0, 1, 0])
        fid = self.api.add_face([p1, p2, p3])
        assert fid in self.api.scene.faces
        assert len(self.api.scene.edges) == 3

    def test_add_face_with_four_points(self):
        p1 = self.api.add_point([0, 0, 0])
        p2 = self.api.add_point([1, 0, 0])
        p3 = self.api.add_point([1, 1, 0])
        p4 = self.api.add_point([0, 1, 0])
        fid = self.api.add_face([p1, p2, p3, p4])
        assert fid in self.api.scene.faces
        assert len(self.api.scene.edges) == 4

    def test_face_creates_edges_automatically(self):
        p1 = self.api.add_point([0, 0, 0])
        p2 = self.api.add_point([1, 0, 0])
        p3 = self.api.add_point([0, 1, 0])
        self.api.add_face([p1, p2, p3])

        # рёбра созданы
        assert len(self.api.scene.edges) == 3

        # грань ссылается на эти рёбра
        for face in self.api.scene.faces.values():
            assert len(face.edge_ids) == 3
