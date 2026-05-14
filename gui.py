"""
3D Editor Koteika Edition GUI - realisation on Dear PyGui
"""

import dearpygui.dearpygui as dpg
import numpy as np
import math

from core import GeometryAPI
from geometry import EPSILON


class ViewState:
    """Данные камеры для поворота и масштабирования"""

    def __init__(self, zoom=120.0):
        self.angle_x = 0.0      # radians
        self.angle_y = 0.0      # radians
        self.angle_z = 0.0      # radians
        self.zoom = zoom
        self.offset_x = 0.0
        self.offset_y = 0.0

    def rotation_matrix(self):
        """Возвращает матрицу вращения 3x3"""
        cx, sx = math.cos(self.angle_x), math.sin(self.angle_x)
        cy, sy = math.cos(self.angle_y), math.sin(self.angle_y)
        cz, sz = math.cos(self.angle_z), math.sin(self.angle_z)

        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        return rz @ ry @ rx


class FaceCulling:
    """Управление видимостью граней, ребер и точек в зависимости от вида камеры"""

    def __init__(self):
        self.backface = True          # Скрыть задние грани
        self.hide_back_edges = True   # Скрыть ребра на задних гранях
        self.hide_back_points = True  # Скрыть точки на задних гранях

    def get_face_normal(self, face, points):
        """Вычисляет нормаль к грани в мировых координатах"""
        vertices = [points[vid].position for vid in face.vertex_ids]
        if len(vertices) < 3:
            return None

        v0, v1, v2 = vertices[:3]
        normal = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normal)

        if norm > EPSILON:
            return normal / norm
        return None

    def is_face_visible(self, face, points, view_matrix) -> bool:
        """Проверяет видимость грани (обращенной к камере)"""
        if not self.backface:
            return True

        normal = self.get_face_normal(face, points)
        if normal is None:
            return True

        # трансформируем нормаль в пространство камеры
        normal_view = view_matrix[:3, :3] @ normal
        return normal_view[2] > 0

    def is_edge_visible(self, edge, points, faces, view_matrix) -> bool:
        """Проверяет видимость ребра (видна хотя бы одна смежная грань)"""
        if not self.hide_back_edges:
            return True

        for face in faces:
            if edge.id in face.edge_ids:
                if self.is_face_visible(face, points, view_matrix):
                    return True
        return False

    def is_point_visible(self, point_id, points, faces, view_matrix) -> bool:
        """Проверяет видимость точки (видна хотя бы одна смежная грань)"""
        if not self.hide_back_points:
            return True

        for face in faces:
            if point_id in face.vertex_ids:
                if self.is_face_visible(face, points, view_matrix):
                    return True
        return False


class GeometryViewer:
    """Main 3D editor window"""

    def __init__(self):
        self.api = GeometryAPI()
        self.view = ViewState(zoom=75.0)
        self.culling = FaceCulling()

        # цвета в формате RGBA для DearPyGui
        self.face_outline = [100, 100, 200, 255]
        self.face_fill = [100, 100, 200, 80]
        self.edge_color = [255, 200, 0, 255]
        self.point_color = [255, 50, 50, 255]

        # статы для мышки
        self.last_mouse = {"x": 0.0, "y": 0.0}

        # стартовая сцена
        self.api.create_cube(center=(-2, 0, 0), size=1.5, name="Cube")
        self.api.create_sphere(center=(2, 0, 0), radius=1.2, name="Sphere")
        self.api.create_pyramid(center=(0, -2, 0), base_size=1.5, height=1.5, name="Pyramid")

    def _project_point(self, pos_3d: np.ndarray, size: tuple) -> tuple:
        """Проецирует 3D точку на 2D, возвращает (координаты, глубина)"""
        rotated = self.view.rotation_matrix() @ pos_3d
        x = rotated[0] * self.view.zoom + size[0] / 2 + self.view.offset_x
        y = -rotated[1] * self.view.zoom + size[1] / 2 + self.view.offset_y
        depth = rotated[2]
        return (float(x), float(y)), depth

    def _draw_face(self, face, size: tuple):
        """Рисует залитую грань с контуром"""
        verts = []
        for vid in face.vertex_ids:
            if vid in self.api.scene.points:
                pos = self.api.scene.points[vid].position
                proj, _ = self._project_point(pos, size)
                verts.append(proj)

        if len(verts) >= 3:
            dpg.draw_polygon(verts, color=self.face_outline, fill=self.face_fill)

    def _draw_edge(self, edge, size: tuple):
        """Рисует ребро как линию"""
        if (edge.point_1_id in self.api.scene.points and edge.point_2_id in self.api.scene.points):
            p1_pos = self.api.scene.points[edge.point_1_id].position
            p2_pos = self.api.scene.points[edge.point_2_id].position

            p1, _ = self._project_point(p1_pos, size)
            p2, _ = self._project_point(p2_pos, size)
            dpg.draw_line(p1, p2, color=self.edge_color, thickness=2)

    def _draw_point(self, point, size: tuple):
        """Рисует точку как кружок"""
        pos, _ = self._project_point(point.position, size)
        dpg.draw_circle(pos, 5, color=self.point_color, fill=self.point_color)

    def render(self, viewport_tag: str, drawlist_tag: str):
        """Основной рендер для сцены"""
        w = dpg.get_item_width(viewport_tag)
        h = dpg.get_item_height(viewport_tag)
        if w <= 0 or h <= 0:
            return

        if dpg.does_item_exist(drawlist_tag):
            dpg.delete_item(drawlist_tag)

        with dpg.drawlist(width=w, height=h, tag=drawlist_tag, parent=viewport_tag):
            # создаём новый контейнер для рисования
            view_matrix = self.view.rotation_matrix()
            view_matrix_4x4 = np.eye(4)
            view_matrix_4x4[:3, :3] = view_matrix

            faces_list = list(self.api.scene.faces.values())
            edges_list = list(self.api.scene.edges.values())
            points_list = list(self.api.scene.points.values())

             # собираем видимые грани с их глубиной для сортировки
            visible_faces = []
            for face in faces_list:
                if self.culling.is_face_visible(face, self.api.scene.points, view_matrix_4x4):
                    center = np.mean([self.api.scene.points[vid].position for vid in face.vertex_ids], axis=0)
                    _, depth = self._project_point(center, (w, h))
                    visible_faces.append((depth, face))

            # сортируем грани по глубине (от дальних к ближним)
            # reverse=True - дальние рисуются первыми, ближние поверх
            visible_faces.sort(key=lambda x: x[0], reverse=True)

            # рисуем грани (от дальних к ближним - правильный порядок)
            for _, face in visible_faces:
                self._draw_face(face, (w, h))

            # рисуем видимые рёбра
            for edge in edges_list:
                if self.culling.is_edge_visible(edge, self.api.scene.points, faces_list, view_matrix_4x4):
                    self._draw_edge(edge, (w, h))

            # рисуем видимые точки
            for point in points_list:
                if self.culling.is_point_visible(point.id, self.api.scene.points, faces_list, view_matrix_4x4):
                    self._draw_point(point, (w, h))

    def run(self):
        """Запускает графический интерфейс"""
        dpg.create_context()  # cоздаём контекст DearPyGui
        dpg.create_viewport(title="Koteika 3D Editor", width=1280, height=720)
        dpg.setup_dearpygui()  # настраиваем DearPyGui

        # главное окно
        with dpg.window(tag="MainWindow", width=1280, height=720,
                        no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.group(horizontal=True):
                # viewport
                with dpg.child_window(tag="Viewport", width=980, height=680, border=True,
                                      no_scrollbar=True, no_scroll_with_mouse=True):
                    dpg.add_text("Left drag - rotate | Shift+Left drag - pan | Wheel - zoom")
                    dpg.add_spacer(height=5)

                # сontrol panel
                with dpg.child_window(width=290, height=680, border=True,
                                      no_scrollbar=True, no_scroll_with_mouse=True):

                    dpg.add_text("Camera Controls", color=(100, 200, 255))
                    dpg.add_separator()

                    dpg.add_text("Rotation X")
                    dpg.add_slider_float(tag="rot_x", default_value=0, min_value=-180, max_value=180,
                                         width=260, callback=lambda s, a: self._update_rotation())

                    dpg.add_text("Rotation Y")
                    dpg.add_slider_float(tag="rot_y", default_value=0, min_value=-180, max_value=180,
                                         width=260, callback=lambda s, a: self._update_rotation())

                    dpg.add_text("Rotation Z")
                    dpg.add_slider_float(tag="rot_z", default_value=0, min_value=-180, max_value=180,
                                         width=260, callback=lambda s, a: self._update_rotation())

                    dpg.add_text("Zoom")
                    dpg.add_slider_float(tag="zoom_slider", default_value=75.0, min_value=20.0, max_value=500.0,
                                         width=260, callback=lambda s, a: setattr(self.view, 'zoom', a))

                    dpg.add_separator()

                    dpg.add_text("Culling", color=(100, 200, 255))
                    dpg.add_checkbox(label="Hide back faces", default_value=True,
                                     callback=lambda s, a: setattr(self.culling, 'backface', a))
                    dpg.add_checkbox(label="Hide back edges", default_value=True,
                                     callback=lambda s, a: setattr(self.culling, 'hide_back_edges', a))
                    dpg.add_checkbox(label="Hide back points", default_value=True,
                                     callback=lambda s, a: setattr(self.culling, 'hide_back_points', a))

                    dpg.add_separator()

                    dpg.add_text("Colors", color=(100, 200, 255))

                    dpg.add_color_edit(tag="face_color_picker", 
                                       default_value=self.face_outline,
                                       width=260, no_inputs=True, label="Face Color",
                                       callback=lambda s, a: self._update_face_color(a))

                    dpg.add_color_edit(tag="edge_color_picker", 
                                       default_value=self.edge_color,
                                       width=260, no_inputs=True, label="Edge Color",
                                       callback=lambda s, a: self._update_edge_color(a))

                    dpg.add_color_edit(tag="point_color_picker", 
                                       default_value=self.point_color,
                                       width=260, no_inputs=True, label="Point Color",
                                       callback=lambda s, a: self._update_point_color(a))

                    dpg.add_separator()

                    dpg.add_text("Objects", color=(100, 200, 255))
                    dpg.add_button(label="Add Cube", width=260,
                                   callback=lambda: self.api.create_cube(center=(0, 0, 0), size=1.0))
                    dpg.add_button(label="Add Pyramid", width=260,
                                   callback=lambda: self.api.create_pyramid(center=(0, 0, 0), base_size=1.0, height=1.0))
                    dpg.add_button(label="Add Sphere", width=260,
                                   callback=lambda: self.api.create_sphere(center=(0, 0, 0), radius=1.0))
                    dpg.add_button(label="Add Icosahedron", width=260,
                                   callback=lambda: self.api.create_icosahedron(center=(0, 0, 0), radius=1.0))
                    dpg.add_button(label="Clear Scene", width=260,
                                   callback=lambda: self.api.clear())

                    dpg.add_separator()

                    self.info_text = dpg.add_text("", color=(150, 150, 150))

        # обработчики мыши
        def on_mouse_move(sender, app_data):
            if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                mx, my = dpg.get_mouse_pos(local=False)
                dx = mx - self.last_mouse["x"]
                dy = my - self.last_mouse["y"]

                shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

                if shift_pressed:
                    self.view.offset_x += dx
                    self.view.offset_y += dy
                else:
                    self.view.angle_y += dx * 0.005
                    self.view.angle_x += dy * 0.005
                    self._sync_sliders()

                self.last_mouse["x"] = mx
                self.last_mouse["y"] = my

        def on_mouse_down(sender, app_data):
            mx, my = dpg.get_mouse_pos(local=False)
            self.last_mouse["x"] = mx
            self.last_mouse["y"] = my

        def on_mouse_wheel(sender, app_data):
            self.view.zoom *= 1.05 if app_data > 0 else 0.95
            self.view.zoom = max(20.0, min(500.0, self.view.zoom))
            dpg.set_value("zoom_slider", self.view.zoom)

        # регистрируем обработчики
        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=on_mouse_move)
            dpg.add_mouse_click_handler(callback=on_mouse_down)
            dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)

        dpg.set_primary_window("MainWindow", True)   # Главное окно
        dpg.show_viewport()  # Показываем окно

        drawlist_tag = "drawlist"

        # главный цикл рендеринга
        while dpg.is_dearpygui_running():
            dpg.set_value(self.info_text, 
                         f"Points: {len(self.api.scene.points)} | "
                         f"Edges: {len(self.api.scene.edges)} | "
                         f"Faces: {len(self.api.scene.faces)}")
            self.render("Viewport", drawlist_tag)
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def _sync_sliders(self):
        """Синхронизирует слайдеры с текущими углами камеры (градусы)"""
        dpg.set_value("rot_x", math.degrees(self.view.angle_x))
        dpg.set_value("rot_y", math.degrees(self.view.angle_y))
        dpg.set_value("rot_z", math.degrees(self.view.angle_z))

    def _update_rotation(self):
        """Обновляет углы камеры из слайдеров (градусы -> радианы)"""
        self.view.angle_x = math.radians(dpg.get_value("rot_x"))
        self.view.angle_y = math.radians(dpg.get_value("rot_y"))
        self.view.angle_z = math.radians(dpg.get_value("rot_z"))

    def _update_face_color(self, color):
        """Обновляет цвет граней"""
        self.face_outline = [color[0], color[1], color[2], 255]
        self.face_fill = [color[0], color[1], color[2], 80]

    def _update_edge_color(self, color):
        """Обновляет цвет рёбер"""
        self.edge_color = [color[0], color[1], color[2], 255]

    def _update_point_color(self, color):
        """Обновляет цвет точек"""
        self.point_color = [color[0], color[1], color[2], 255]


def main():
    viewer = GeometryViewer()
    viewer.run()


if __name__ == "__main__":
    main()