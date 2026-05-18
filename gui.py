"""
3D Editor Koteika Edition GUI - realisation on Dear PyGui
"""

import dearpygui.dearpygui as dpg
from pathlib import Path
import numpy as np
import math


from geometry import EPSILON
from core import GeometryAPI


class ViewState:
    """Данные камеры для поворота и масштабирования"""

    def __init__(self, zoom=75.0):
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
        self.backface = True          # скрыть задние грани
        self.hide_back_edges = True   # скрыть ребра на задних гранях
        self.hide_back_points = True  # скрыть точки на задних гранях

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

        # цвета в формате RGBA для DearPyGui (0-255)
        self.face_outline = [100, 100, 200, 255]
        self.face_fill = [100, 100, 200, 80]
        self.edge_color = [255, 200, 0, 255]
        self.point_color = [255, 50, 50, 255]
        self.section_color = [0, 255, 180, 255]

        # статы для мышки
        self.last_mouse = {"x": 0.0, "y": 0.0}
        self.viewport_rect = None
        self.drawlist_rect = None
        self.last_draw_size = (980, 680)

        self.dragging_point = False
        self.selected_point_id = None
        self.point_hit_radius = 16.0

        # сохранение
        self.status_message = ""
        self.scene_path = Path("scene.json")
        self.png_path = Path("scene.png")

        # сечения
        self.section_enabled = False
        self.section_normal = np.array([0.0, 0.0, 1.0])
        self.section_offset = 0.0

        # снимок состояния точек для оптимизации обновления UI
        self.points_snapshot = None

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
        radius = 7 if point.id == self.selected_point_id else 5
        color = [255, 255, 255, 255] if point.id == self.selected_point_id else self.point_color
        if point.id == self.selected_point_id:
            dpg.draw_circle(pos, self.point_hit_radius, color=[255, 255, 255, 90], thickness=1)
        dpg.draw_circle(pos, radius, color=color, fill=color)

    def _draw_section_segment(self, segment, size: tuple):
        start, end = segment
        p1, _ = self._project_point(start, size)
        p2, _ = self._project_point(end, size)
        dpg.draw_line(p1, p2, color=self.section_color, thickness=3)

    def _update_viewport_rect(self):
        """Обновляет прямоугольник области вьюпорта"""
        try:
            pos = dpg.get_item_pos("Viewport")
            width = dpg.get_item_width("Viewport")
            height = dpg.get_item_height("Viewport")
            if pos and width > 0 and height > 0:
                self.viewport_rect = (pos[0], pos[1], pos[0] + width, pos[1] + height)
        except:
            self.viewport_rect = None

    def _is_mouse_over_viewport(self) -> bool:
        """Проверяет находится ли мышь над областью вьюпорта"""
        if self.viewport_rect is None:
            self._update_viewport_rect()

        if self.viewport_rect is None:
            return True

        min_x, min_y, max_x, max_y = self.viewport_rect
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)

        return (min_x <= mouse_x <= max_x and min_y <= mouse_y <= max_y)

    def _update_drawlist_rect(self, drawlist_tag: str):
        """Обновляет прямоугольник области рисования (drawlist)"""
        try:
            if dpg.does_item_exist(drawlist_tag):
                pos = dpg.get_item_pos(drawlist_tag)
                width = dpg.get_item_width(drawlist_tag)
                height = dpg.get_item_height(drawlist_tag)
                if pos and width > 0 and height > 0:
                    self.drawlist_rect = (pos[0], pos[1], pos[0] + width, pos[1] + height)
        except:
            self.drawlist_rect = None

    def _get_mouse_draw_pos(self):
        """Преобразует глобальные координаты мыши в локальные координаты drawlist"""
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        rect = self.drawlist_rect or self.viewport_rect
        if rect is None:
            return mouse_x, mouse_y
        return mouse_x - rect[0], mouse_y - rect[1]

    def _is_control_pressed(self) -> bool:
        """Проверяет зажат ли любой Ctrl"""
        keys = [getattr(dpg, "mvKey_LControl", None), getattr(dpg, "mvKey_RControl", None)]
        return any(key is not None and dpg.is_key_down(key) for key in keys)

    def _find_nearest_point(self, mouse_x: float, mouse_y: float, max_distance: float = None):
        """Находит ближайшую к курсору точку в пределах радиуса"""
        if max_distance is None:
            max_distance = self.point_hit_radius

        nearest_id = None
        nearest_distance = max_distance

        for point in self.api.scene.points.values():
            pos, _ = self._project_point(point.position, self.last_draw_size)
            distance = math.hypot(pos[0] - mouse_x, pos[1] - mouse_y)
            if distance <= nearest_distance:
                nearest_id = point.id
                nearest_distance = distance

        return nearest_id

    def _move_selected_point_by_screen_delta(self, dx: float, dy: float):
        """Перемещает выбранную точку на экранное смещение"""

        # пользователь перетаскивает точку мышью, это перемещение приисходит в экранных координатах
        # функция преобразует это смещение обратно в 3D-пространство с учетом текущего поворота камеры

        if self.selected_point_id not in self.api.scene.points:
            return

        # преобразуем экранное смещение в смещение в пространстве камеры
        delta_view = np.array([dx / self.view.zoom, -dy / self.view.zoom, 0.0])

        # переводим в мировые координаты (обратная матрица поворота)
        delta_world = self.view.rotation_matrix().T @ delta_view

        self.api.move_point(self.selected_point_id, delta_world)
        self._sync_point_inputs()

    # комбобокс - это название елемента UI - выподающего списка
    # он сочетает в себе два элемента:
    #  -> поле ввода
    #  -> выпадающий список
    def _point_items(self):
        """Формирует список строк для комбобокса точек"""
        return [
            f"{point_id}: {point.position[0]:.2f}, {point.position[1]:.2f}, {point.position[2]:.2f}"
            for point_id, point in sorted(self.api.scene.points.items())
        ]

    def _refresh_point_selector(self, force: bool = False):
        """Обновляет список точек в комбобоксе и синхронизирует выделение"""
        if not dpg.does_item_exist("point_selector"):
            return

        # создаем снимок текущего состояния точек для сравнения
        snapshot = tuple(
            (point_id, tuple(np.round(point.position, 4)))
            for point_id, point in sorted(self.api.scene.points.items())
        )

        # обновляем UI только если состояние изменилось
        sync_inputs = force or snapshot != self.points_snapshot
        if sync_inputs:
            items = self._point_items()
            dpg.configure_item("point_selector", items=items)
            self.points_snapshot = snapshot

        # если выбранная точка была удалена, выбираем первую доступную
        if self.selected_point_id is not None and self.selected_point_id not in self.api.scene.points:
            self.selected_point_id = next(iter(sorted(self.api.scene.points)), None)
            sync_inputs = True

        # обновляем отображение в комбобоксе
        if self.selected_point_id is None:
            dpg.set_value("point_selector", "")
        else:
            point = self.api.scene.points[self.selected_point_id]
            dpg.set_value(
                "point_selector",
                f"{point.id}: {point.position[0]:.2f}, {point.position[1]:.2f}, {point.position[2]:.2f}"
            )

        if sync_inputs:
            self._sync_point_inputs()

    def _select_point_from_combo(self, value: str):
        """Обработчик выбора точки из комбобокса"""
        if not value:
            self.selected_point_id = None
            return
        # извлекаем ID точки из строки "123: 1.00, 2.00, 3.00"
        self.selected_point_id = int(str(value).split(":", 1)[0])
        self._sync_point_inputs()

    def _sync_point_inputs(self):
        """Синхронизирует поля ввода координат с выбранной точкой"""
        if self.selected_point_id is None or self.selected_point_id not in self.api.scene.points:
            # очищаем поля if точки нет
            if dpg.does_item_exist("point_x"):
                dpg.set_value("point_x", 0.0)
                dpg.set_value("point_y", 0.0)
                dpg.set_value("point_z", 0.0)
            return

        if not dpg.does_item_exist("point_x"):
            return

        point = self.api.scene.points[self.selected_point_id]
        dpg.set_value("point_x", float(point.position[0]))
        dpg.set_value("point_y", float(point.position[1]))
        dpg.set_value("point_z", float(point.position[2]))

    def _apply_point_inputs(self):
        """Применяет координаты из полей ввода к выбранной точке"""
        if self.selected_point_id is None or self.selected_point_id not in self.api.scene.points:
            self.status_message = "No point selected"
            return

        position = np.array([
            dpg.get_value("point_x"),
            dpg.get_value("point_y"),
            dpg.get_value("point_z"),
        ], dtype=float)

        self.api.move_point_to(self.selected_point_id, position)
        self.status_message = f"Point {self.selected_point_id} moved"
        self._refresh_point_selector(force=True)

# =======================| Сечения |===========================

    def _update_section(self):
        """Обновляет параметры сечения из UI"""
        self.section_enabled = bool(dpg.get_value("section_enabled"))
        self.section_normal = np.array([
            dpg.get_value("section_nx"),
            dpg.get_value("section_ny"),
            dpg.get_value("section_nz"),
        ], dtype=float)
        self.section_offset = float(dpg.get_value("section_offset"))

    def _get_section_segments(self):
        """Возвращает список отрезков (начало, конец) пересечения плоскости с геометрией"""
        if self.section_enabled is False:
            return []
        try:
            return self.api.section_by_plane(self.section_normal, self.section_offset)
        except ValueError as error:
            self.status_message = str(error)
            return []

# =======================| Сохранение Чтение json |===========================

    def _path_from_input(self, tag: str, default_path: Path) -> Path:
        """Получает путь из текстового поля ввода"""
        value = str(dpg.get_value(tag) or "").strip()
        return Path(value) if value else default_path

    def _save_scene_to_path(self, path: Path):
        """Сохраняет сцену"""
        if not path.suffix:
            path = path.with_suffix(".json")  # добавляем .json if нет расширения
        try:
            self.api.save(path)
        except OSError as error:  # ошибка записи на диск
            self.status_message = str(error)
            return
        self._set_scene_path(path)  # обновляем путь в UI
        self.status_message = f"Scene saved: {path}"


    def _save_scene(self):
        """Сохраняет сцену в корневую директорию проекта (из текстового поля)"""
        path = self._path_from_input("scene_path", self.scene_path)
        try:
            self.api.save(path)
        except OSError as error:
            self.status_message = str(error)
            return
        self.scene_path = path
        self.status_message = f"Scene saved: {path}"

    def _load_scene(self):
        """Загружает сцену из файла, лежащего в корневой директории проекта"""
        path = self._path_from_input("scene_path", self.scene_path)
        try:
            self.api.load(path)
        except (OSError, ValueError) as error:
            self.status_message = str(error)
            return
        self.scene_path = path
        self.status_message = f"Scene loaded: {path}"
        self._refresh_point_selector(force=True)

# =======================| Экспорт png  |===========================

    def _export_png(self):
        """Экспорт в корневую директорию проекта"""
        path = self._path_from_input("png_path", self.png_path)
        width = max(1, int(self.last_draw_size[0]))
        height = max(1, int(self.last_draw_size[1]))
        try:
            image = self._render_to_image(width, height)
        except RuntimeError as error:
            self.status_message = str(error)
            return
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        self.png_path = path
        self.status_message = f"PNG exported: {path}"

    def _render_to_image(self, width: int, height: int):
        """Рендерит текущую 3D сцену в изображение PNG с помощью библиотеки Pillow"""
        try:
            from PIL import Image, ImageDraw
        except ImportError as error:
            raise RuntimeError("Pillow is required for PNG export") from error

        image = Image.new("RGBA", (width, height), (25, 25, 30, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        size = (width, height)

        # получаем матрицу поворота камеры (3x3)
        # создаем матрицу 4x4 для работы с 3D трансформациям
        view_matrix = self.view.rotation_matrix()
        view_matrix_4x4 = np.eye(4)
        view_matrix_4x4[:3, :3] = view_matrix

        faces_list = list(self.api.scene.faces.values())
        edges_list = list(self.api.scene.edges.values())
        points_list = list(self.api.scene.points.values())

        # вычисляем глубину граней в 3-ёх меронм пространтстве
        visible_faces = []
        for face in faces_list:
            if self.culling.is_face_visible(face, self.api.scene.points, view_matrix_4x4):
                center = np.mean([self.api.scene.points[vid].position for vid in face.vertex_ids], axis=0)
                _, depth = self._project_point(center, size)
                visible_faces.append((depth, face))

        visible_faces.sort(key=lambda x: x[0], reverse=True)

        # рисуем
        for _, face in visible_faces:
            verts = []
            for vid in face.vertex_ids:
                if vid in self.api.scene.points:
                    proj, _ = self._project_point(self.api.scene.points[vid].position, size)
                    verts.append(proj)
            if len(verts) >= 3:
                draw.polygon(verts, outline=tuple(self.face_outline), fill=tuple(self.face_fill))

        for edge in edges_list:
            if self.culling.is_edge_visible(edge, self.api.scene.points, faces_list, view_matrix_4x4):
                if edge.point_1_id in self.api.scene.points and edge.point_2_id in self.api.scene.points:
                    p1, _ = self._project_point(self.api.scene.points[edge.point_1_id].position, size)
                    p2, _ = self._project_point(self.api.scene.points[edge.point_2_id].position, size)
                    draw.line([p1, p2], fill=tuple(self.edge_color), width=2)

        for segment in self._get_section_segments():
            p1, _ = self._project_point(segment[0], size)
            p2, _ = self._project_point(segment[1], size)
            draw.line([p1, p2], fill=tuple(self.section_color), width=3)

        for point in points_list:
            if self.culling.is_point_visible(point.id, self.api.scene.points, faces_list, view_matrix_4x4):
                pos, _ = self._project_point(point.position, size)
                radius = 7 if point.id == self.selected_point_id else 5
                color = (255, 255, 255, 255) if point.id == self.selected_point_id else tuple(self.point_color)
                if point.id == self.selected_point_id:
                    hit_radius = int(self.point_hit_radius)
                    draw.ellipse(
                        [pos[0] - hit_radius, pos[1] - hit_radius, pos[0] + hit_radius, pos[1] + hit_radius],
                        outline=(255, 255, 255, 90)
                    )
                draw.ellipse(
                    [pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius],
                    outline=color,
                    fill=color
                )

        return image

# =======================| Render  |===========================

    def render(self, viewport_tag: str, drawlist_tag: str):
        """Основной рендер для сцены"""
        w = dpg.get_item_width(viewport_tag)
        h = dpg.get_item_height(viewport_tag)
        if w <= 0 or h <= 0:
            return

        # Обновляем rect вьюпорта
        self._update_viewport_rect()

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

            # рисуем сечения
            for segment in self._get_section_segments():
                self._draw_section_segment(segment, (w, h))

            # рисуем видимые точки
            for point in points_list:
                if self.culling.is_point_visible(point.id, self.api.scene.points, faces_list, view_matrix_4x4):
                    self._draw_point(point, (w, h))

            self._update_drawlist_rect(drawlist_tag)    

# =======================| GUI  |===========================

    def run(self):
        """Запускает графический интерфейс"""
        dpg.create_context()  # cоздаём контекст DearPyGui
        dpg.create_viewport(title="Koteika 3D Editor", width=1400, height=750)
        dpg.setup_dearpygui()  # настраиваем DearPyGui

        # главное окно
        with dpg.window(tag="MainWindow", width=1400, height=720,
                        no_scrollbar=True, no_scroll_with_mouse=True):

            with dpg.group(horizontal=True):
                # viewport
                with dpg.child_window(tag="Viewport", width=980, height=680, border=True,
                                      no_scrollbar=True, no_scroll_with_mouse=True):
                    dpg.add_text("Left drag - rotate | Shift+Left drag - pan | Wheel - zoom")
                    dpg.add_spacer(height=5)

                # сontrol panel
                with dpg.child_window(width=500, height=680, border=True):

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

                    # конвертируем цвета из 0-255 в 0-1 для color_edit
                    dpg.add_color_edit(tag="face_color_picker", 
                                       default_value=self.face_outline[:3],
                                       width=260, no_inputs=True, label="Face Color",
                                       callback=lambda s, a: self._update_face_color(a))

                    dpg.add_color_edit(tag="edge_color_picker", 
                                       default_value=self.edge_color[:3],
                                       width=260, no_inputs=True, label="Edge Color",
                                       callback=lambda s, a: self._update_edge_color(a))

                    dpg.add_color_edit(tag="point_color_picker", 
                                       default_value=self.point_color[:3],
                                       width=260, no_inputs=True, label="Point Color",
                                       callback=lambda s, a: self._update_point_color(a))

                    # color picker для сечений
                    dpg.add_color_edit(tag="section_color_picker",
                                    default_value=self.section_color[:3],
                                    width=260, no_inputs=True, label="Section Color",
                                    callback=lambda s, a: self._update_section_color(a))

                    dpg.add_separator()

                    dpg.add_text("Objects", color=(100, 200, 255))

                    dpg.add_button(label="Add Cube", width=260,
                                   callback=lambda: self.api.create_cube(center=(0, 0, 0), size=1.0))
                    dpg.add_button(label="Add Pyramid", width=260,
                                   callback=lambda: self.api.create_pyramid(center=(0, 0, 0), base_size=1.0, height=1.0))
                    dpg.add_button(label="Add Sphere", width=260,
                                   callback=lambda: self.api.create_sphere(center=(0, 0, 0), radius=1.0))
                    dpg.add_button(label="Add Sphere quad", width=260,
                                   callback=lambda: self.api.create_sphere_quads(center=(0, 0, 0), radius=1.0))
                    dpg.add_button(label="Add Icosahedron", width=260,
                                   callback=lambda: self.api.create_icosahedron(center=(0, 0, 0), radius=1.0))
                    dpg.add_button(label="Clear Scene", width=260,
                                   callback=lambda: self.api.clear())

                    dpg.add_separator()

                    dpg.add_text("Point Move", color=(100, 200, 255))
                    dpg.add_combo(tag="point_selector", items=[], width=260,
                                  callback=lambda s, a: self._select_point_from_combo(a))
                    dpg.add_input_float(tag="point_x", label="X", width=200, step=0.1)
                    dpg.add_input_float(tag="point_y", label="Y", width=200, step=0.1)
                    dpg.add_input_float(tag="point_z", label="Z", width=200, step=0.1)
                    dpg.add_slider_float(tag="point_hit_radius", label="Hit Radius", default_value=self.point_hit_radius,
                                         min_value=6.0, max_value=40.0, width=200,
                                         callback=lambda s, a: setattr(self, 'point_hit_radius', a))
                    dpg.add_button(label="Apply Point Position", width=260,
                                   callback=lambda: self._apply_point_inputs())

                    dpg.add_separator()

                    dpg.add_text("Section", color=(100, 200, 255))
                    dpg.add_checkbox(tag="section_enabled", label="Show section", default_value=False,
                                     callback=lambda s, a: self._update_section())
                    dpg.add_input_float(tag="section_nx", label="Nx", default_value=0.0, width=200, step=0.1,
                                        callback=lambda s, a: self._update_section())
                    dpg.add_input_float(tag="section_ny", label="Ny", default_value=0.0, width=200, step=0.1,
                                        callback=lambda s, a: self._update_section())
                    dpg.add_input_float(tag="section_nz", label="Nz", default_value=1.0, width=200, step=0.1,
                                        callback=lambda s, a: self._update_section())
                    dpg.add_input_float(tag="section_offset", label="Offset", default_value=0.0, width=200, step=0.1,
                                        callback=lambda s, a: self._update_section())

                    dpg.add_separator()

                    dpg.add_text("Save / Export", color=(100, 200, 255))
                    dpg.add_input_text(tag="scene_path", default_value=str(self.scene_path), width=260)
                    dpg.add_button(label="Save Scene", width=260, callback=lambda: self._save_scene())
                    dpg.add_button(label="Load Scene", width=260, callback=lambda: self._load_scene())
                    dpg.add_input_text(tag="png_path", default_value=str(self.png_path), width=260)
                    dpg.add_button(label="Export PNG", width=260, callback=lambda: self._export_png())

                    dpg.add_separator()

                    self.info_text = dpg.add_text("", color=(150, 150, 150))

        # обработчики мыши с проверкой над вьюпортом
        def on_mouse_move(sender, app_data):
            if self._is_mouse_over_viewport():
                if dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                    mx, my = dpg.get_mouse_pos(local=False)
                    dx = mx - self.last_mouse["x"]
                    dy = my - self.last_mouse["y"]

                    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
                    control_pressed = self._is_control_pressed() 

                    if control_pressed and self.selected_point_id is not None:
                        self._move_selected_point_by_screen_delta(dx, dy)
                    elif shift_pressed:
                        self.view.offset_x += dx
                        self.view.offset_y += dy
                    else:
                        self.view.angle_y += dx * 0.005
                        self.view.angle_x += dy * 0.005
                        self._sync_sliders()

                    self.last_mouse["x"] = mx
                    self.last_mouse["y"] = my

        def on_mouse_down(sender, app_data):
            if self._is_mouse_over_viewport():
                mx, my = dpg.get_mouse_pos(local=False)
                self.last_mouse["x"] = mx
                self.last_mouse["y"] = my

                draw_x, draw_y = self._get_mouse_draw_pos()
                point_id = self._find_nearest_point(draw_x, draw_y)
                if point_id is not None:
                    self.selected_point_id = point_id
                    self._refresh_point_selector(force=True)

        def on_mouse_wheel(sender, app_data):
            if self._is_mouse_over_viewport():
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
            self._refresh_point_selector()
            section_count = len(self._get_section_segments())
            dpg.set_value(self.info_text, 
                         f"Points: {len(self.api.scene.points)} | "
                         f"Edges: {len(self.api.scene.edges)} | "
                         f"Faces: {len(self.api.scene.faces)} | "
                         f"Sections: {section_count}\n"
                         f"{self.status_message}")
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
        # color приходит в формате RGB в диапазоне 0-1
        r, g, b = [int(c * 255) for c in color[:3]]
        self.face_outline = [r, g, b, 255]
        self.face_fill = [r, g, b, 80]

    def _update_edge_color(self, color):
        """Обновляет цвет рёбер"""
        # color приходит в формате RGB в диапазоне 0-1
        r, g, b = [int(c * 255) for c in color[:3]]
        self.edge_color = [r, g, b, 255]

    def _update_point_color(self, color):
        """Обновляет цвет точек"""
        # color приходит в формате RGB в диапазоне 0-1
        r, g, b = [int(c * 255) for c in color[:3]]
        self.point_color = [r, g, b, 255]

    def _update_section_color(self, color):
        """Обновляет цвет линий сечений"""
        # color приходит в формате RGB в диапазоне 0-1
        r, g, b = [int(c * 255) for c in color[:3]]
        self.section_color = [r, g, b, 255]


def main():
    viewer = GeometryViewer()
    viewer.run()


if __name__ == "__main__":
    main()
