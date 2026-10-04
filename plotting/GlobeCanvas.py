# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter


class GlobeCanvas:

    def __init__(
        self,
        earth_radius=6378.145,
        earth_texture="bluemarble-2048.png",
        cloud_texture="clouds_2048.png",
        asset_logo="ASSETLOGO.png",
        alabama_logo="ALABAMALOGO.png",
        title="Earth Globe",
        window_size=(1500, 950),
        show_clouds=True,
        show_atmosphere=True,
        show_equator=True,
        show_controls=True,
        show_sim_time=True,
        show_asset_logo=True,
        show_alabama_logo=True,
        background="black",
        auto_setup=True,
    ):

        self.earth_radius = float(earth_radius)
        self.earth_texture = earth_texture
        self.cloud_texture = cloud_texture
        self.asset_logo = asset_logo
        self.alabama_logo = alabama_logo
        self.title = title
        self.window_size = window_size
        self.show_clouds = bool(show_clouds)
        self.show_atmosphere = bool(show_atmosphere)
        self.show_equator = bool(show_equator)
        self.show_controls = bool(show_controls)
        self.show_sim_time = bool(show_sim_time)
        self.show_asset_logo = bool(show_asset_logo)
        self.show_alabama_logo = bool(show_alabama_logo)
        self.background = background

        self.plotter = BackgroundPlotter(title=self.title, window_size=self.window_size)
        self.plotter.set_background(self.background)

        self.earth_mesh = None
        self.earth_actor = None
        self.cloud_mesh = None
        self.cloud_actor = None
        self.atmosphere_actor = None
        self.equator_actor = None

        self.trajectory_actors = []
        self.marker_actors = []
        self.vector_actors = []

        self.legend_labels = {}

        self.controls_text_actor = None
        self.sim_time_text_actor = None
        self.asset_logo_widget = None
        self.alabama_logo_widget = None

        self.animation_callback = None
        self.animation_update_function = None
        self.animation_reset_function = None
        self.animation_time = 0.0
        self.animation_time_scale = 1.0
        self.animation_min_time_scale = 0.1
        self.animation_max_time_scale = 5000.0
        self.animation_running = False
        self.animation_last_wall_time = None

        if auto_setup:
            self._setup_scene()
            self._setup_canvas_overlays()
            self._setup_keyboard_controls()
            self._setup_asset_logo()
            self._setup_alabama_logo()

    def _resolve_texture(self, filename):
        if filename is None:
            return None

        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename

        candidates = [
            filename,
            os.path.join(os.path.dirname(__file__), filename),
            os.path.join(os.getcwd(), filename),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return None

    def _create_equirectangular_sphere(self, radius, n_lon=360, n_lat=180):
        longitude = np.linspace(0.0, 2.0 * np.pi, n_lon + 1)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_lat + 1)

        n_points = (n_lon + 1) * (n_lat + 1)

        points = np.empty((n_points, 3), dtype=np.float64)
        texture_coordinates = np.empty((n_points, 2), dtype=np.float64)

        index = 0

        for j, lat in enumerate(latitude):
            cos_lat = np.cos(lat)
            sin_lat = np.sin(lat)
            v = j / n_lat

            for i, lon in enumerate(longitude):
                u = i / n_lon

                x = radius * cos_lat * np.cos(lon)
                y = radius * cos_lat * np.sin(lon)
                z = radius * sin_lat

                points[index] = [x, y, z]
                texture_coordinates[index] = [u, v]

                index += 1

        faces = []
        row_size = n_lon + 1

        for j in range(n_lat):
            for i in range(n_lon):
                p0 = j * row_size + i
                p1 = p0 + 1
                p2 = (j + 1) * row_size + i
                p3 = p2 + 1

                faces.extend([4, p0, p1, p3, p2])

        faces = np.asarray(faces, dtype=np.int64)

        sphere = pv.PolyData(points, faces)
        sphere.active_texture_coordinates = texture_coordinates

        return sphere

    def _setup_scene(self):
        self._add_earth()
        self._add_clouds()
        self._add_atmosphere()
        self._add_equator()
        self._configure_lighting()
        self._configure_camera()

    def _add_earth(self):
        self.earth_mesh = self._create_equirectangular_sphere(self.earth_radius)

        texture_path = self._resolve_texture(self.earth_texture)

        if texture_path is not None:
            texture = pv.read_texture(texture_path)
            self.earth_actor = self.plotter.add_mesh(self.earth_mesh, texture=texture, smooth_shading=True)
        else:
            self.earth_actor = self.plotter.add_mesh(self.earth_mesh, smooth_shading=True)

    def _add_clouds(self):
        if not self.show_clouds:
            return

        cloud_radius = self.earth_radius * 1.002
        self.cloud_mesh = self._create_equirectangular_sphere(cloud_radius)

        texture_path = self._resolve_texture(self.cloud_texture)

        if texture_path is None:
            return

        texture = pv.read_texture(texture_path)
        self.cloud_actor = self.plotter.add_mesh(self.cloud_mesh, texture=texture, opacity=0.35, smooth_shading=True)

    def _add_atmosphere(self):
        if not self.show_atmosphere:
            return

        atmosphere = pv.Sphere(radius=self.earth_radius * 1.015, theta_resolution=180, phi_resolution=90)
        self.atmosphere_actor = self.plotter.add_mesh(atmosphere, color="lightskyblue", opacity=0.06, smooth_shading=True)

    def _add_equator(self):
        if not self.show_equator:
            return

        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        radius = self.earth_radius * 1.003

        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))

        self.equator_actor = self.plotter.add_lines(points, width=1.5, color="white", connected=True)

    def _configure_lighting(self):
        self.plotter.remove_all_lights()

        main_light = pv.Light(position=(30000.0, -20000.0, 20000.0), focal_point=(0.0, 0.0, 0.0), intensity=1.8)
        fill_light = pv.Light(position=(-25000.0, 15000.0, 10000.0), focal_point=(0.0, 0.0, 0.0), intensity=0.20)

        self.plotter.add_light(main_light)
        self.plotter.add_light(fill_light)

    def _configure_camera(self):
        distance = self.earth_radius * 3.0

        self.plotter.camera.position = (distance, -distance, distance * 0.75)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    def _setup_canvas_overlays(self):
        if self.show_controls:
            controls = "SPACE  Pause / Resume\nr      Reset\n+ / =  Increase Speed\n- / _  Decrease Speed"
            self.controls_text_actor = self.plotter.add_text(controls, position=(25, 25), font_size=9, color="white", shadow=True)

        if self.show_sim_time:
            self.sim_time_text_actor = self.plotter.add_text(self._simulation_status_text(), position=(1110, 25), font_size=9, color="white", shadow=True)

    def _setup_asset_logo(self):
        if not self.show_asset_logo:
            return

        logo_path = self._resolve_texture(self.asset_logo)

        if logo_path is None:
            print(f"ASSET logo not found: {self.asset_logo}")
            return

        try:
            print(f"ASSET logo loaded: {logo_path}")
            print(f"ASSET logo exists: {os.path.isfile(logo_path)}")

            self.asset_logo_widget = self.plotter.add_logo_widget(logo_path, position=(0.78, 0.80), size=(0.25, 0.18), opacity=1.0)

            print("ASSET logo widget created successfully.")

        except Exception as exc:
            print(f"Failed to load ASSET logo: {exc}")
            self.asset_logo_widget = None

    def _setup_alabama_logo(self):
        if not self.show_alabama_logo:
            return

        logo_path = self._resolve_texture(self.alabama_logo)

        if logo_path is None:
            print(f"University of Alabama logo not found: {self.alabama_logo}")
            return

        try:
            print(f"University of Alabama logo loaded: {logo_path}")
            print(f"University of Alabama logo exists: {os.path.isfile(logo_path)}")

            self.alabama_logo_widget = self.plotter.add_logo_widget(logo_path, position=(0.78, 0.7), size=(0.25, 0.08), opacity=1.0)

            print("University of Alabama logo widget created successfully.")

        except Exception as exc:
            print(f"Failed to load University of Alabama logo: {exc}")
            self.alabama_logo_widget = None

    def _setup_keyboard_controls(self):
        self.plotter.add_key_event("space", self.toggle_animation)
        self.plotter.add_key_event("r", self.reset_animation)
        self.plotter.add_key_event("=", self.increase_animation_speed)
        self.plotter.add_key_event("+", self.increase_animation_speed)
        self.plotter.add_key_event("-", self.decrease_animation_speed)
        self.plotter.add_key_event("_", self.decrease_animation_speed)

    def _simulation_status_text(self):
        status = "RUNNING" if self.animation_running else "PAUSED"

        return f"Simulation Time\nt = {self.animation_time:,.2f} s\nSpeed = {self.animation_time_scale:.2f}x\nStatus = {status}"

    def _update_simulation_time_text(self):
        if self.sim_time_text_actor is None:
            return

        text = self._simulation_status_text()

        if hasattr(self.sim_time_text_actor, "input"):
            self.sim_time_text_actor.input = text
        elif hasattr(self.sim_time_text_actor, "SetInput"):
            self.sim_time_text_actor.SetInput(text)
        elif hasattr(self.sim_time_text_actor, "SetText"):
            self.sim_time_text_actor.SetText(2, text)

        if hasattr(self.sim_time_text_actor, "Modified"):
            self.sim_time_text_actor.Modified()

    def _register_legend_label(self, label, color):
        if label is None:
            return

        self.legend_labels[label] = color

    def plot_trajectory(self, trajectory, color="white", width=3.0, label=None, normalized=False, Lstar=None, opacity=1.0):
        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2 or trajectory.shape[1] < 3:
            raise ValueError("trajectory must be an Nx3 or NxM array")

        points = trajectory[:, :3].copy()

        if normalized:
            if Lstar is None:
                raise ValueError("Lstar must be supplied when normalized=True")

            points *= float(Lstar)

        polyline = pv.lines_from_points(points)
        actor = self.plotter.add_mesh(polyline, color=color, line_width=width, opacity=opacity)

        self.trajectory_actors.append(actor)
        self._register_legend_label(label, color)

        return actor

    def plot_point(self, point, color="white", size=12.0, label=None):
        point = np.asarray(point, dtype=float).reshape(3)

        actor = self.plotter.add_points(point.reshape(1, 3), color=color, point_size=size, render_points_as_spheres=True)

        self.marker_actors.append(actor)
        self._register_legend_label(label, color)

        return actor

    def plot_startpoint(self, trajectory, color="green", size=12.0, label="Start"):
        trajectory = np.asarray(trajectory, dtype=float)
        return self.plot_point(trajectory[0, :3], color=color, size=size, label=label)

    def plot_endpoint(self, trajectory, color="red", size=12.0, label="End"):
        trajectory = np.asarray(trajectory, dtype=float)
        return self.plot_point(trajectory[-1, :3], color=color, size=size, label=label)

    def plot_vector(self, origin, vector, color="yellow", scale=1.0, width=3.0, label=None):
        origin = np.asarray(origin, dtype=float).reshape(3)
        vector = np.asarray(vector, dtype=float).reshape(3)

        mesh = pv.Arrow(start=origin, direction=vector, scale=float(scale))
        actor = self.plotter.add_mesh(mesh, color=color, line_width=width)

        self.vector_actors.append(actor)
        self._register_legend_label(label, color)

        return actor

    def create_batched_trajectories(self, trajectories):
        all_points = []
        all_lines = []
        point_offset = 0

        for trajectory in trajectories:
            trajectory = np.asarray(trajectory, dtype=float)

            if trajectory.ndim != 2 or trajectory.shape[1] < 3:
                raise ValueError("Each trajectory must be an Nx3 or NxM array")

            points = trajectory[:, :3]
            n_points = len(points)

            all_points.append(points)

            line = np.concatenate(([n_points], np.arange(point_offset, point_offset + n_points)))
            all_lines.append(line)

            point_offset += n_points

        if not all_points:
            return pv.PolyData()

        points = np.vstack(all_points)
        lines = np.concatenate(all_lines)

        return pv.PolyData(points, lines=lines)

    def plot_batched_trajectories(self, trajectories, color="white", width=2.0, label=None, opacity=1.0):
        mesh = self.create_batched_trajectories(trajectories)
        actor = self.plotter.add_mesh(mesh, color=color, line_width=width, opacity=opacity)

        self.trajectory_actors.append(actor)
        self._register_legend_label(label, color)

        return actor

    def create_point_mesh(self, positions):
        positions = np.asarray(positions, dtype=float)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape Nx3")

        return pv.PolyData(positions)

    def plot_points(self, positions, color="white", size=8.0, label=None):
        mesh = self.create_point_mesh(positions)

        actor = self.plotter.add_mesh(mesh, color=color, point_size=size, render_points_as_spheres=True)

        self.marker_actors.append(actor)
        self._register_legend_label(label, color)

        return mesh, actor

    def update_points(self, positions):
        positions = np.asarray(positions, dtype=float)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape Nx3")

        for actor in self.marker_actors:
            mesh = actor.mapper.dataset

            if mesh.n_points == positions.shape[0]:
                mesh.points = positions
                mesh.GetPoints().Modified()
                mesh.Modified()

    def add_legend(self):
        if not self.legend_labels:
            return

        labels = [[label, color] for label, color in self.legend_labels.items()]

        try:
            self.plotter.add_legend(labels, bcolor=(0.05, 0.05, 0.05, 0.75), face="none", loc="lower_right", size=(0.20, 0.18))
        except (ValueError, TypeError):
            try:
                self.plotter.add_legend(labels, bcolor=(0.05, 0.05, 0.05, 0.75), loc="lower_right")
            except Exception:
                pass

    def add_animation(self, update_function, interval=30, time_scale=60.0, min_time_scale=0.1, max_time_scale=5000.0, reset_function=None, start=True):
        self.animation_update_function = update_function
        self.animation_reset_function = reset_function
        self.animation_time_scale = float(time_scale)
        self.animation_min_time_scale = float(min_time_scale)
        self.animation_max_time_scale = float(max_time_scale)
        self.animation_running = bool(start)
        self.animation_last_wall_time = time.perf_counter()

        if self.animation_callback is not None:
            try:
                self.plotter.clear_callback(self.animation_callback)
            except Exception:
                pass

        self.animation_callback = self.plotter.add_callback(self._animation_tick, interval=interval)
        self._update_simulation_time_text()

    def _animation_tick(self, *args):
        if not self.animation_running:
            self.animation_last_wall_time = time.perf_counter()
            return

        now = time.perf_counter()

        if self.animation_last_wall_time is None:
            self.animation_last_wall_time = now
            return

        dt_wall = now - self.animation_last_wall_time
        self.animation_last_wall_time = now

        self.animation_time += dt_wall * self.animation_time_scale

        if self.animation_update_function is not None:
            self.animation_update_function(self.animation_time)

        self._update_simulation_time_text()

        try:
            self.plotter.render()
        except Exception:
            pass

    def pause_animation(self):
        self.animation_running = False
        self.animation_last_wall_time = time.perf_counter()
        self._update_simulation_time_text()

    def resume_animation(self):
        self.animation_running = True
        self.animation_last_wall_time = time.perf_counter()
        self._update_simulation_time_text()

    def toggle_animation(self):
        if self.animation_running:
            self.pause_animation()
        else:
            self.resume_animation()

    def reset_animation(self):
        self.animation_time = 0.0
        self.animation_last_wall_time = time.perf_counter()

        if self.animation_reset_function is not None:
            self.animation_reset_function()

        if self.animation_update_function is not None:
            self.animation_update_function(self.animation_time)

        self._update_simulation_time_text()

        try:
            self.plotter.render()
        except Exception:
            pass

    def set_animation_speed(self, time_scale):
        self.animation_time_scale = float(np.clip(time_scale, self.animation_min_time_scale, self.animation_max_time_scale))
        self._update_simulation_time_text()

    def increase_animation_speed(self):
        self.set_animation_speed(self.animation_time_scale * 2.0)

    def decrease_animation_speed(self):
        self.set_animation_speed(self.animation_time_scale / 2.0)

    def stop_animation(self):
        self.animation_running = False

        if self.animation_callback is not None:
            try:
                self.plotter.clear_callback(self.animation_callback)
            except Exception:
                pass

            self.animation_callback = None

        self._update_simulation_time_text()

    def frame(self, padding=1.25):
        bounds = np.asarray(self.earth_mesh.bounds, dtype=float)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        center = np.array([
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ])

        radius = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
        distance = radius * float(padding) * 2.0

        direction = np.array([1.0, -1.0, 0.75])
        direction /= np.linalg.norm(direction)

        self.plotter.camera.focal_point = center
        self.plotter.camera.position = center + direction * distance
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    def set_camera(self, position, focal_point=(0.0, 0.0, 0.0), view_up=(0.0, 0.0, 1.0)):
        self.plotter.camera.position = tuple(position)
        self.plotter.camera.focal_point = tuple(focal_point)
        self.plotter.camera.up = tuple(view_up)

    def set_camera_distance(self, distance, focal_point=(0.0, 0.0, 0.0)):
        direction = np.asarray(self.plotter.camera.position, dtype=float)
        direction -= np.asarray(focal_point, dtype=float)

        norm = np.linalg.norm(direction)

        if norm == 0.0:
            direction = np.array([1.0, -1.0, 0.75])
            norm = np.linalg.norm(direction)

        direction /= norm

        self.plotter.camera.focal_point = tuple(focal_point)
        self.plotter.camera.position = tuple(np.asarray(focal_point) + direction * float(distance))

    def clear_trajectories(self):
        for actor in self.trajectory_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.trajectory_actors = []

    def clear_markers(self):
        for actor in self.marker_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.marker_actors = []

    def clear_vectors(self):
        for actor in self.vector_actors:
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.vector_actors = []

    def clear(self):
        self.clear_trajectories()
        self.clear_markers()
        self.clear_vectors()
        self.legend_labels = {}

    def show(self):
        self.plotter.app.exec_()