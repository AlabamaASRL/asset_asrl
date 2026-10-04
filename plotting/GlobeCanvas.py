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
        show_stars=True,
        show_reference_axis=True,
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
        self.background = background

        self.show_clouds = show_clouds
        self.show_atmosphere = show_atmosphere
        self.show_equator = show_equator
        self.show_stars = show_stars
        self.show_reference_axis = show_reference_axis
        self.show_controls = show_controls
        self.show_sim_time = show_sim_time
        self.show_asset_logo = show_asset_logo
        self.show_alabama_logo = show_alabama_logo

        self.plotter = BackgroundPlotter(title=self.title, window_size=self.window_size)
        self.plotter.set_background(self.background)

        self.earth_actor = None
        self.cloud_actor = None
        self.atmosphere_actor = None
        self.equator_actor = None
        self.reference_axis_actor = None
        self.star_actor = None

        self.trajectory_actors = []
        self.marker_actors = []
        self.vector_actors = []
        self.last_point_actor = None

        self.spacecraft_actor = None
        self.spacecraft_panel_actors = []
        self.spacecraft_position = np.zeros(3, dtype=float)

        self.legend_entries = []

        self.controls_text_actor = None
        self.sim_time_text_actor = None
        self.title_text_actor = None
        self.asset_logo_actor = None
        self.alabama_logo_actor = None

        self.animation_callback = None
        self.animation_timer = None
        self.animation_running = False
        self.simulation_time = 0.0
        self.animation_time_scale = 60.0
        self.animation_min_time_scale = 0.1
        self.animation_max_time_scale = 5000.0
        self._last_animation_wall_time = None
        self.reset_callback = None

        self.camera_follow = False
        self.follow_position = np.zeros(3, dtype=float)
        self.follow_camera_offset = np.array([3.0, -3.0, 1.5], dtype=float)
        self.follow_distance_scale = 1.0

        if auto_setup:
            self._setup_scene()

    def _setup_scene(self):

        if self.show_stars:
            self._add_stars()

        self._add_earth()

        if self.show_clouds:
            self._add_clouds()

        if self.show_atmosphere:
            self._add_atmosphere()

        if self.show_equator:
            self._add_equator()

        if self.show_reference_axis:
            self._add_reference_axis()

        self._configure_lighting()
        self._configure_camera()
        self._add_overlay()
        self._register_keyboard_controls()

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

    def _create_textured_sphere(self, radius, n_latitude=90, n_longitude=180):

        longitude = np.linspace(0.0, 2.0 * np.pi, n_longitude + 1)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_latitude + 1)

        lon_grid, lat_grid = np.meshgrid(longitude, latitude)

        x = radius * np.cos(lat_grid) * np.cos(lon_grid)
        y = radius * np.cos(lat_grid) * np.sin(lon_grid)
        z = radius * np.sin(lat_grid)

        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        faces = []

        for i in range(n_latitude):
            for j in range(n_longitude):
                p0 = i * (n_longitude + 1) + j
                p1 = p0 + 1
                p2 = p1 + (n_longitude + 1)
                p3 = p0 + (n_longitude + 1)
                faces.extend([4, p0, p1, p2, p3])

        faces = np.asarray(faces, dtype=np.int64)

        sphere = pv.PolyData(points, faces)

        u = lon_grid / (2.0 * np.pi)
        v = (lat_grid + 0.5 * np.pi) / np.pi

        sphere.active_texture_coordinates = np.column_stack((u.ravel(), v.ravel()))

        return sphere

    def _add_earth(self):

        earth_mesh = self._create_textured_sphere(self.earth_radius, n_latitude=90, n_longitude=180)
        texture_path = self._resolve_texture(self.earth_texture)

        if texture_path is not None:
            texture = pv.read_texture(texture_path)
            self.earth_actor = self.plotter.add_mesh(earth_mesh, texture=texture, smooth_shading=True, name="Earth")
        else:
            self.earth_actor = self.plotter.add_mesh(earth_mesh, color="royalblue", smooth_shading=True, name="Earth")

    def _add_clouds(self):

        cloud_mesh = self._create_textured_sphere(self.earth_radius * 1.002, n_latitude=90, n_longitude=180)
        texture_path = self._resolve_texture(self.cloud_texture)

        if texture_path is None:
            return

        texture = pv.read_texture(texture_path)
        self.cloud_actor = self.plotter.add_mesh(cloud_mesh, texture=texture, opacity=0.35, smooth_shading=True, name="Clouds")

    def _add_atmosphere(self):

        atmosphere = pv.Sphere(radius=self.earth_radius * 1.015, theta_resolution=180, phi_resolution=90)
        self.atmosphere_actor = self.plotter.add_mesh(atmosphere, color="lightskyblue", opacity=0.06, smooth_shading=True, name="Atmosphere")

    def _add_equator(self):

        theta = np.linspace(0.0, 2.0 * np.pi, 720)
        radius = self.earth_radius * 1.003
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))

        polyline = pv.lines_from_points(points)
        self.equator_actor = self.plotter.add_mesh(polyline, color="white", line_width=2.0, name="Equator")

    def _add_reference_axis(self):

        radius = self.earth_radius * 1.08
        points = np.array([[0.0, 0.0, -radius], [0.0, 0.0, radius]])

        axis = pv.lines_from_points(points)
        self.reference_axis_actor = self.plotter.add_mesh(axis, color="gray", line_width=2.0, name="Earth Reference Axis")

    def _add_stars(self):

        rng = np.random.default_rng(12345)
        number_of_stars = 1800

        radius = rng.uniform(5.5 * self.earth_radius, 9.0 * self.earth_radius, number_of_stars)
        phi = rng.uniform(0.0, 2.0 * np.pi, number_of_stars)
        cos_theta = rng.uniform(-1.0, 1.0, number_of_stars)
        theta = np.arccos(cos_theta)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        points = np.column_stack((x, y, z))
        stars = pv.PolyData(points)

        self.star_actor = self.plotter.add_mesh(stars, color="white", point_size=2.0, render_points_as_spheres=True, name="Stars")

    def _configure_lighting(self):

        main_light = pv.Light(position=(30000.0, -20000.0, 20000.0), focal_point=(0.0, 0.0, 0.0))
        main_light.intensity = 1.8
        self.plotter.add_light(main_light)

        fill_light = pv.Light(position=(-25000.0, 15000.0, 10000.0), focal_point=(0.0, 0.0, 0.0))
        fill_light.intensity = 0.20
        self.plotter.add_light(fill_light)

    def _configure_camera(self):

        distance = self.earth_radius * 3.0

        self.plotter.camera.position = (distance, -distance, 0.75 * distance)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    def _add_overlay(self):

        self.title_text_actor = self.plotter.add_text(
            "ASSET Mission Analysis",
            position=(25, 20),
            font_size=18,
            color="white",
        )

        if self.show_controls:

            controls = (
                "SPACE  Pause / Resume\n"
                "R      Reset\n"
                "RIGHT  Increase Speed\n"
                "LEFT   Decrease Speed\n"
                "F      Follow Spacecraft"
            )

            self.controls_text_actor = self.plotter.add_text(controls, position=(25, 60), font_size=11, color="white")

        if self.show_sim_time:

            self.sim_time_text_actor = self.plotter.add_text(
                "SIMULATION\n"
                "t = 0.000 s\n"
                "Speed = 60.0x\n"
                "Status = RUNNING\n"
                "Camera = FIXED",
                position=(1110, 25),
                font_size=11,
                color="white",
            )

        if self.show_asset_logo:

            logo_path = self._resolve_texture(self.asset_logo)

            if logo_path is not None:
                self.asset_logo_actor = self.plotter.add_logo_widget(
                    logo_path,
                    position=(0.78, 0.80),
                    size=(0.25, 0.18),
                    opacity=1.0,
                )

        if self.show_alabama_logo:

            logo_path = self._resolve_texture(self.alabama_logo)

            if logo_path is not None:
                self.alabama_logo_actor = self.plotter.add_logo_widget(
                    logo_path,
                    position=(0.78, 0.70),
                    size=(0.25, 0.08),
                    opacity=1.0,
                )

    def _register_keyboard_controls(self):

        self.plotter.add_key_event("space", self.toggle_animation)
        self.plotter.add_key_event("r", self.reset_animation)
        self.plotter.add_key_event("Right", self._increase_speed_key)
        self.plotter.add_key_event("Left", self._decrease_speed_key)
        self.plotter.add_key_event("f", self.toggle_camera_follow)

    def _increase_speed_key(self):

        old_speed = self.animation_time_scale
        self.increase_animation_speed()
        print(f"Animation speed: {old_speed:g}x -> {self.animation_time_scale:g}x")

    def _decrease_speed_key(self):

        old_speed = self.animation_time_scale
        self.decrease_animation_speed()
        print(f"Animation speed: {old_speed:g}x -> {self.animation_time_scale:g}x")

    def _update_simulation_time_text(self):

        if self.sim_time_text_actor is None:
            return

        status = "RUNNING" if self.animation_running else "PAUSED"
        camera_mode = "FOLLOW" if self.camera_follow else "FIXED"

        text = (
            "SIMULATION\n"
            f"t = {self.simulation_time:.3f} s\n"
            f"Speed = {self.animation_time_scale:g}x\n"
            f"Status = {status}\n"
            f"Camera = {camera_mode}"
        )

        try:
            self.sim_time_text_actor.input = text
        except Exception:
            try:
                self.sim_time_text_actor.SetInput(text)
            except Exception:
                try:
                    self.sim_time_text_actor.SetText(2, text)
                except Exception:
                    pass

    def plot_trajectory(self, trajectory, color="white", width=3.0, label=None, normalized=False, Lstar=None, opacity=1.0):

        trajectory = np.asarray(trajectory)

        if trajectory.ndim != 2:
            raise ValueError("Trajectory must be a 2D array.")

        if trajectory.shape[1] < 3:
            raise ValueError("Trajectory must contain at least 3 columns.")

        points = trajectory[:, :3].astype(float)

        if normalized:

            if Lstar is None:
                raise ValueError("Lstar must be supplied when normalized=True.")

            points = points * float(Lstar)

        polyline = pv.lines_from_points(points)
        actor = self.plotter.add_mesh(polyline, color=color, line_width=width, opacity=opacity)

        self.trajectory_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def plot_point(self, position, color="white", size=12.0, label=None):

        position = np.asarray(position, dtype=float)
        mesh = pv.PolyData(position.reshape(1, 3))
        actor = self.plotter.add_mesh(mesh, color=color, point_size=size, render_points_as_spheres=True)

        self.marker_actors.append(actor)
        self.last_point_actor = actor

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def plot_startpoint(self, trajectory, color="green", size=12.0, label="Start"):

        trajectory = np.asarray(trajectory)
        return self.plot_point(trajectory[0, :3], color=color, size=size, label=label)

    def plot_endpoint(self, trajectory, color="red", size=12.0, label="End"):

        trajectory = np.asarray(trajectory)
        return self.plot_point(trajectory[-1, :3], color=color, size=size, label=label)

    def plot_vector(self, origin, vector, color="yellow", scale=1.0, width=3.0, label=None):

        origin = np.asarray(origin, dtype=float)
        vector = np.asarray(vector, dtype=float)
        magnitude = np.linalg.norm(vector)

        if magnitude == 0.0:
            return None

        arrow = pv.Arrow(start=origin, direction=vector, scale=scale)
        actor = self.plotter.add_mesh(arrow, color=color, line_width=width)

        self.vector_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def create_batched_trajectories(self, trajectories):

        trajectories = np.asarray(trajectories, dtype=float)

        if trajectories.ndim != 3:
            raise ValueError("Trajectories must have shape (N_trajectories, N_points, 3).")

        points = []
        lines = []
        point_offset = 0

        for trajectory in trajectories:

            n_points = trajectory.shape[0]
            points.extend(trajectory)
            lines.extend([n_points, *range(point_offset, point_offset + n_points)])
            point_offset += n_points

        return pv.PolyData(
            np.asarray(points, dtype=float),
            lines=np.asarray(lines, dtype=np.int64),
        )

    def plot_batched_trajectories(self, trajectories, color="white", width=2.0, opacity=1.0, label=None):

        mesh = self.create_batched_trajectories(trajectories)
        actor = self.plotter.add_mesh(mesh, color=color, line_width=width, opacity=opacity)

        self.trajectory_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def create_point_mesh(self, points):

        points = np.asarray(points, dtype=float)
        return pv.PolyData(points)

    def plot_points(self, points, color="white", size=8.0, label=None):

        mesh = self.create_point_mesh(points)
        actor = self.plotter.add_mesh(mesh, color=color, point_size=size, render_points_as_spheres=True)

        self.marker_actors.append(actor)
        self.last_point_actor = actor

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def update_points(self, points_or_actor, points=None):

        if points is None:
            actor = self.last_point_actor
            points = points_or_actor
        else:
            actor = points_or_actor

        if actor is None:
            raise RuntimeError("No point actor is available for update. Call plot_points() first.")

        points = np.asarray(points, dtype=float)

        if points.ndim != 2:
            raise ValueError("Points must be a 2D array.")

        if points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3).")

        try:
            mesh = actor.mapper.dataset
        except Exception:
            mesh = actor

        mesh.points = points
        mesh.GetPoints().Modified()
        mesh.Modified()

    def plot_spacecraft(self, position, scale=None, color="white", panel_color="silver", label="Spacecraft"):

        position = np.asarray(position, dtype=float)

        if scale is None:
            scale = self.earth_radius * 0.035

        scale = float(scale)

        body_mesh = pv.Cube(center=position, x_length=scale, y_length=0.65 * scale, z_length=0.65 * scale)
        self.spacecraft_actor = self.plotter.add_mesh(body_mesh, color=color, smooth_shading=True)

        panel_offset = 0.85 * scale
        panel_width = 0.70 * scale
        panel_height = 0.25 * scale

        left_center = position + np.array([-panel_offset, 0.0, 0.0])
        right_center = position + np.array([panel_offset, 0.0, 0.0])

        left_panel = pv.Cube(center=left_center, x_length=panel_width, y_length=panel_height, z_length=0.10 * scale)
        right_panel = pv.Cube(center=right_center, x_length=panel_width, y_length=panel_height, z_length=0.10 * scale)

        left_actor = self.plotter.add_mesh(left_panel, color=panel_color, smooth_shading=True)
        right_actor = self.plotter.add_mesh(right_panel, color=panel_color, smooth_shading=True)

        self.spacecraft_panel_actors = [left_actor, right_actor]
        self.spacecraft_position = position.copy()
        self.follow_position = position.copy()

        if label is not None:
            self._register_legend_label(label, color)

        return self.spacecraft_actor

    def update_spacecraft(self, position):

        position = np.asarray(position, dtype=float)

        self.spacecraft_position = position.copy()
        self.follow_position = position.copy()

        if self.spacecraft_actor is None:
            return

        body_mesh = self.spacecraft_actor.mapper.dataset
        old_body_center = np.mean(body_mesh.points, axis=0)
        delta = position - old_body_center

        body_mesh.points = body_mesh.points + delta
        body_mesh.GetPoints().Modified()
        body_mesh.Modified()

        for actor in self.spacecraft_panel_actors:

            panel_mesh = actor.mapper.dataset
            panel_mesh.points = panel_mesh.points + delta
            panel_mesh.GetPoints().Modified()
            panel_mesh.Modified()

        if self.camera_follow:
            self._update_follow_camera(position)

    def _update_follow_camera(self, position):

        position = np.asarray(position, dtype=float)
        offset = np.asarray(self.follow_camera_offset, dtype=float)
        offset_norm = np.linalg.norm(offset)

        if offset_norm == 0.0:
            offset = np.array([3.0, -3.0, 1.5], dtype=float)
            offset_norm = np.linalg.norm(offset)

        direction = offset / offset_norm
        camera_distance = self.earth_radius * 1.5 * self.follow_distance_scale
        camera_position = position + direction * camera_distance

        self.plotter.camera.position = camera_position
        self.plotter.camera.focal_point = position
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    def enable_camera_follow(self, position=None):

        if position is not None:

            position = np.asarray(position, dtype=float)
            self.follow_position = position.copy()
            self.spacecraft_position = position.copy()

        else:

            self.follow_position = self.spacecraft_position.copy()

        self.camera_follow = True
        self._update_follow_camera(self.follow_position)
        self._update_simulation_time_text()
        self.plotter.render()

        print("Camera follow: ON")

    def disable_camera_follow(self):

        self.camera_follow = False
        self._configure_camera()
        self._update_simulation_time_text()
        self.plotter.render()

        print("Camera follow: OFF")

    def toggle_camera_follow(self):

        if self.camera_follow:
            self.disable_camera_follow()
        else:
            self.enable_camera_follow()

    def set_follow_camera_offset(self, offset):

        offset = np.asarray(offset, dtype=float)

        if offset.shape != (3,):
            raise ValueError("Camera offset must contain exactly three values.")

        self.follow_camera_offset = offset

        if self.camera_follow:
            self._update_follow_camera(self.follow_position)
            self.plotter.render()

    def set_follow_distance(self, scale):

        self.follow_distance_scale = max(float(scale), 0.01)

        if self.camera_follow:
            self._update_follow_camera(self.follow_position)
            self.plotter.render()

    def _register_legend_label(self, label, color):
        self.legend_entries.append((label, color))

    def add_legend(self):

        if not self.legend_entries:
            return None

        try:
            return self.plotter.add_legend(bcolor=(0.05, 0.05, 0.05), border=True, size=(0.20, 0.20), loc="lower right")
        except Exception:
            try:
                return self.plotter.add_legend(loc="lower right")
            except Exception:
                return None

    def add_animation(
        self,
        update_function,
        interval=30,
        time_scale=60.0,
        min_time_scale=0.1,
        max_time_scale=5000.0,
        reset_function=None,
        start=True,
    ):

        self.animation_callback = update_function
        self.animation_time_scale = float(time_scale)
        self.animation_min_time_scale = float(min_time_scale)
        self.animation_max_time_scale = float(max_time_scale)
        self.reset_callback = reset_function
        self.simulation_time = 0.0
        self.animation_running = bool(start)
        self._last_animation_wall_time = time.perf_counter()

        if self.animation_timer is not None:

            try:
                self.animation_timer.stop()
            except Exception:
                pass

            self.animation_timer = None

        self.animation_timer = self.plotter.add_callback(self._animation_tick, interval)

        if self.animation_callback is not None:

            try:
                self.animation_callback(self.simulation_time)
            except Exception as exc:
                print("Animation initialization error:")
                print(repr(exc))
                self.animation_running = False

        self._update_simulation_time_text()
        self.plotter.render()

        return self.animation_timer

    def _animation_tick(self, *_):

        if self.animation_callback is None:
            return

        now = time.perf_counter()

        if self._last_animation_wall_time is None:
            self._last_animation_wall_time = now

        wall_dt = now - self._last_animation_wall_time
        self._last_animation_wall_time = now
        wall_dt = float(np.clip(wall_dt, 0.0, 0.25))

        if not self.animation_running:
            self._update_simulation_time_text()
            return

        simulation_dt = wall_dt * self.animation_time_scale
        self.simulation_time += simulation_dt

        try:
            self.animation_callback(self.simulation_time)

        except Exception as exc:

            print("\nAnimation callback error:")
            print(repr(exc))
            print("Animation paused.")

            self.animation_running = False
            self._update_simulation_time_text()
            return

        self._update_simulation_time_text()
        self.plotter.render()

    def pause_animation(self):

        self.animation_running = False
        self._update_simulation_time_text()

    def resume_animation(self):

        self.animation_running = True
        self._last_animation_wall_time = time.perf_counter()
        self._update_simulation_time_text()

    def toggle_animation(self):

        if self.animation_running:
            self.pause_animation()
            print("Animation paused.")
        else:
            self.resume_animation()
            print("Animation resumed.")

    def reset_animation(self):

        self.simulation_time = 0.0
        self.animation_running = True
        self._last_animation_wall_time = time.perf_counter()

        if self.reset_callback is not None:

            try:
                self.reset_callback()
            except Exception as exc:
                print("Reset callback error:", repr(exc))

        self._update_simulation_time_text()
        self.plotter.render()

        print("Simulation reset.")

    def set_animation_speed(self, time_scale):

        self.animation_time_scale = float(np.clip(time_scale, self.animation_min_time_scale, self.animation_max_time_scale))
        self._update_simulation_time_text()

    def increase_animation_speed(self):
        self.set_animation_speed(self.animation_time_scale * 2.0)

    def decrease_animation_speed(self):
        self.set_animation_speed(self.animation_time_scale / 2.0)

    def stop_animation(self):

        if self.animation_timer is not None:

            try:
                self.animation_timer.stop()
            except Exception:
                pass

        self.animation_timer = None
        self.animation_callback = None
        self.animation_running = False

        self._update_simulation_time_text()

    def frame(self, padding=1.25):

        radius = self.earth_radius * float(padding)
        distance = radius * 2.5

        self.plotter.camera.position = (distance, -distance, 0.75 * distance)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    def set_camera(self, position, focal_point=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)):

        self.plotter.camera.position = np.asarray(position, dtype=float)
        self.plotter.camera.focal_point = np.asarray(focal_point, dtype=float)
        self.plotter.camera.up = np.asarray(up, dtype=float)

    def set_camera_distance(self, distance):

        direction = np.array([1.0, -1.0, 0.75], dtype=float)
        direction /= np.linalg.norm(direction)

        self.plotter.camera.position = direction * float(distance)
        self.plotter.camera.focal_point = (0.0, 0.0, 0.0)

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
        self.last_point_actor = None

    def clear_vectors(self):

        for actor in self.vector_actors:

            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.vector_actors = []

    def clear_spacecraft(self):

        if self.spacecraft_actor is not None:

            try:
                self.plotter.remove_actor(self.spacecraft_actor)
            except Exception:
                pass

        for actor in self.spacecraft_panel_actors:

            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass

        self.spacecraft_actor = None
        self.spacecraft_panel_actors = []

    def clear(self):

        self.clear_trajectories()
        self.clear_markers()
        self.clear_vectors()
        self.clear_spacecraft()

    def show(self):
        self.plotter.app.exec_()