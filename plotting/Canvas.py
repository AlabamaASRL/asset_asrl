# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

from Earth import Earth
from Spacecraft import Spacecraft


class Canvas:
    """PyVista-based visualization canvas for celestial bodies and spacecraft."""

    def __init__(self, earth=None, title="ASSET Mission Analysis", window_size=(1500, 950), show_stars=True, show_controls=True, show_sim_time=True, show_asset_logo=True, show_alabama_logo=True, asset_logo="ASSETLOGO.png", alabama_logo="ALABAMALOGO.png", background="black"):
        self.title = title
        self.window_size = window_size
        self.background = background
        self.show_stars = show_stars
        self.show_controls = show_controls
        self.show_sim_time = show_sim_time
        self.show_asset_logo = show_asset_logo
        self.show_alabama_logo = show_alabama_logo
        self.asset_logo = asset_logo
        self.alabama_logo = alabama_logo

        self.earth = earth if earth is not None else Earth()
        self.bodies = [self.earth]
        self.plotter = BackgroundPlotter(title=self.title, window_size=self.window_size)
        self.plotter.set_background(self.background)

        self.star_actor = None
        self.spacecraft = None
        self.trajectory_actors = []
        self.marker_actors = []
        self.vector_actors = []
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

        self._setup_scene()

    #%% Scene setup

    def _setup_scene(self):
        if self.show_stars:
            self._add_stars()

        for body in self.bodies:
            body.add_to_canvas(self)

        self._configure_lighting()
        self._configure_camera()
        self._add_overlay()
        self._register_keyboard_controls()

    def _resolve_texture(self, filename):
        if filename is None:
            return None

        candidates = [filename, os.path.join(os.path.dirname(__file__), filename), os.path.join(os.getcwd(), filename)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    #%% Celestial bodies

    def add_body(self, body):
        if body in self.bodies:
            return getattr(body, "actor", None)

        self.bodies.append(body)
        return body.add_to_canvas(self)

    def remove_body(self, body):
        if body not in self.bodies:
            return

        if hasattr(body, "remove_from_canvas"):
            body.remove_from_canvas()
        else:
            for attribute in ("actor", "cloud_actor", "atmosphere_actor", "equator_actor", "reference_axis_actor"):
                actor = getattr(body, attribute, None)
                if actor is not None:
                    self.plotter.remove_actor(actor)

        self.bodies.remove(body)

    #%% Stars and lighting

    def _add_stars(self):
        rng = np.random.default_rng(12345)
        count = 1800
        reference_radius = max((body.radius for body in self.bodies), default=6378.145)
        radius = rng.uniform(5.5 * reference_radius, 9.0 * reference_radius, count)
        phi = rng.uniform(0.0, 2.0 * np.pi, count)
        cos_theta = rng.uniform(-1.0, 1.0, count)
        theta = np.arccos(cos_theta)

        points = np.column_stack((radius * np.sin(theta) * np.cos(phi), radius * np.sin(theta) * np.sin(phi), radius * np.cos(theta)))
        self.star_actor = self.plotter.add_mesh(pv.PolyData(points), color="white", point_size=2.0, render_points_as_spheres=True, name="Stars")

    def _configure_lighting(self):
        main_light = pv.Light(position=(30000.0, -20000.0, 20000.0), focal_point=(0.0, 0.0, 0.0))
        main_light.intensity = 1.8
        self.plotter.add_light(main_light)

        fill_light = pv.Light(position=(-25000.0, 15000.0, 10000.0), focal_point=(0.0, 0.0, 0.0))
        fill_light.intensity = 0.2
        self.plotter.add_light(fill_light)

    #%% Camera

    def _get_reference_radius(self):
        return max((body.radius for body in self.bodies), default=6378.145)

    def _configure_camera(self):
        distance = 3.0 * self._get_reference_radius()
        self.set_camera_position((distance, -distance, 0.75 * distance), focal_point=(0.0, 0.0, 0.0))

    def set_camera_position(self, position, focal_point=None, up=(0.0, 0.0, 1.0)):
        self.plotter.camera.position = np.asarray(position, dtype=float)
        if focal_point is not None:
            self.plotter.camera.focal_point = np.asarray(focal_point, dtype=float)
        self.plotter.camera.up = up

    def reset_camera(self):
        self._configure_camera()

    def toggle_camera_follow(self):
        self.camera_follow = not self.camera_follow
        if self.camera_follow:
            if self.spacecraft is not None:
                self.follow_position = self.spacecraft.position.copy()
            self._update_follow_camera(self.follow_position)
        self._update_simulation_time_text()

    def _update_follow_camera(self, position):
        position = np.asarray(position, dtype=float)
        self.follow_position = position.copy()
        offset = self.follow_camera_offset * self._get_reference_radius() * 3.0 * self.follow_distance_scale
        self.plotter.camera.position = position + offset
        self.plotter.camera.focal_point = position
        self.plotter.camera.up = (0.0, 0.0, 1.0)

    #%% On-screen text and logos

    def _add_overlay(self):
        self.title_text_actor = self.plotter.add_text("ASSET Mission Analysis", position=(25, 20), font_size=18, color="white")

        if self.show_controls:
            controls = "SPACE  Pause / Resume\nR      Reset\nRIGHT  Increase Speed\nLEFT   Decrease Speed\nF      Follow Spacecraft"
            self.controls_text_actor = self.plotter.add_text(controls, position=(25, 60), font_size=11, color="white")

        if self.show_sim_time:
            self.sim_time_text_actor = self.plotter.add_text("", position=(1110, 25), font_size=11, color="white")
            self._update_simulation_time_text()

        if self.show_asset_logo:
            path = self._resolve_texture(self.asset_logo)
            if path is not None:
                self.asset_logo_actor = self.plotter.add_logo_widget(path, position=(0.78, 0.80), size=(0.25, 0.18), opacity=1.0)

        if self.show_alabama_logo:
            path = self._resolve_texture(self.alabama_logo)
            if path is not None:
                self.alabama_logo_actor = self.plotter.add_logo_widget(path, position=(0.78, 0.70), size=(0.25, 0.08), opacity=1.0)

    def _update_simulation_time_text(self):
        if self.sim_time_text_actor is None:
            return

        status = "RUNNING" if self.animation_running else "PAUSED"
        camera_mode = "FOLLOW" if self.camera_follow else "FIXED"
        text = f"SIMULATION\n t = {self.simulation_time:.3f} s\n Speed = {self.animation_time_scale:g}x\n Status = {status}\n Camera = {camera_mode}"

        try:
            self.sim_time_text_actor.SetInput(text)
        except AttributeError:
            self.sim_time_text_actor.input = text

    #%% Keyboard controls

    def _register_keyboard_controls(self):
        self.plotter.add_key_event("space", self.toggle_animation)
        self.plotter.add_key_event("r", self.reset_animation)
        self.plotter.add_key_event("Right", self.increase_animation_speed)
        self.plotter.add_key_event("Left", self.decrease_animation_speed)
        self.plotter.add_key_event("f", self.toggle_camera_follow)

    #%% Spacecraft

    def add_spacecraft(self, spacecraft):
        if not isinstance(spacecraft, Spacecraft):
            raise TypeError("spacecraft must be a Spacecraft instance.")

        if self.spacecraft is not None:
            self.remove_spacecraft()

        self.spacecraft = spacecraft
        spacecraft.add_to_canvas(self)
        self.follow_position = spacecraft.position.copy()
        self._register_legend_label(spacecraft.name, spacecraft.body_color)
        return spacecraft

    def remove_spacecraft(self):
        if self.spacecraft is None:
            return

        self.spacecraft.remove_from_canvas()
        self.spacecraft = None

    def clear_spacecraft(self):
        self.remove_spacecraft()

    def update_spacecraft(self, position):
        if self.spacecraft is None:
            return

        self.spacecraft.update_position(np.asarray(position, dtype=float))
        self.follow_position = self.spacecraft.position.copy()

        if self.camera_follow:
            self._update_follow_camera(self.spacecraft.position)

    #%% Trajectories, points, and vectors

    def plot_trajectory(self, trajectory, color="white", width=3.0, label=None, normalized=False, Lstar=None, opacity=1.0):
        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2 or trajectory.shape[1] < 3:
            raise ValueError("Trajectory must be a 2D array with at least 3 columns.")

        points = trajectory[:, :3]
        if normalized:
            if Lstar is None:
                raise ValueError("Lstar must be supplied when normalized=True.")
            points = points * float(Lstar)

        actor = self.plotter.add_mesh(pv.lines_from_points(points), color=color, line_width=width, opacity=opacity)
        self.trajectory_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def plot_point(self, position, color="white", size=12.0, label=None):
        mesh = pv.PolyData(np.asarray(position, dtype=float).reshape(1, 3))
        actor = self.plotter.add_mesh(mesh, color=color, point_size=size, render_points_as_spheres=True)
        self.marker_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    def plot_startpoint(self, trajectory, color="green", size=12.0, label="Start"):
        return self.plot_point(np.asarray(trajectory)[0, :3], color=color, size=size, label=label)

    def plot_endpoint(self, trajectory, color="red", size=12.0, label="End"):
        return self.plot_point(np.asarray(trajectory)[-1, :3], color=color, size=size, label=label)

    def plot_vector(self, origin, vector, color="yellow", scale=1.0, width=3.0, label=None):
        origin = np.asarray(origin, dtype=float)
        vector = np.asarray(vector, dtype=float)

        if np.linalg.norm(vector) == 0.0:
            return None

        actor = self.plotter.add_mesh(pv.Arrow(start=origin, direction=vector, scale=scale), color=color, line_width=width)
        self.vector_actors.append(actor)

        if label is not None:
            self._register_legend_label(label, color)

        return actor

    #%% Legend

    def _register_legend_label(self, label, color):
        entry = (str(label), color)
        if entry not in self.legend_entries:
            self.legend_entries.append(entry)
            self._update_legend()

    def _update_legend(self):
        if self.legend_entries:
            self.plotter.add_legend(self.legend_entries, bcolor="black", border=True, size=(0.20, 0.20))

    #%% Animation

    def add_animation(self, update_function, interval=30, time_scale=60.0):
        self.animation_callback = update_function
        self.animation_time_scale = float(time_scale)
        self.animation_running = True
        self._last_animation_wall_time = time.perf_counter()

        # BackgroundPlotter's Qt timer drives the callback repeatedly.
        self.animation_timer = self.plotter.add_callback(self._animation_timer_callback, interval=int(interval))
        self._update_simulation_time_text()

    def _animation_timer_callback(self):
        if not self.animation_running:
            return

        current_wall_time = time.perf_counter()

        if self._last_animation_wall_time is None:
            self._last_animation_wall_time = current_wall_time
            return

        dt_wall = current_wall_time - self._last_animation_wall_time
        self._last_animation_wall_time = current_wall_time
        self.simulation_time += dt_wall * self.animation_time_scale

        if self.animation_callback is not None:
            self.animation_callback(self.simulation_time)

        if self.camera_follow and self.spacecraft is not None:
            self._update_follow_camera(self.spacecraft.position)

        self._update_simulation_time_text()
        self.plotter.render()

    def toggle_animation(self):
        self.animation_running = not self.animation_running
        self._last_animation_wall_time = time.perf_counter() if self.animation_running else None
        self._update_simulation_time_text()

    def reset_animation(self):
        self.simulation_time = 0.0
        self._last_animation_wall_time = time.perf_counter()

        if self.reset_callback is not None:
            self.reset_callback()
        elif self.animation_callback is not None:
            self.animation_callback(0.0)

        self._update_simulation_time_text()
        self.plotter.render()

    def increase_animation_speed(self):
        self.animation_time_scale = min(self.animation_time_scale * 2.0, self.animation_max_time_scale)
        self._update_simulation_time_text()

    def decrease_animation_speed(self):
        self.animation_time_scale = max(self.animation_time_scale / 2.0, self.animation_min_time_scale)
        self._update_simulation_time_text()

    #%% Clear and show

    def clear_trajectories(self):
        for actor in self.trajectory_actors:
            self.plotter.remove_actor(actor)
        self.trajectory_actors.clear()

    def clear_markers(self):
        for actor in self.marker_actors:
            self.plotter.remove_actor(actor)
        self.marker_actors.clear()

    def clear_vectors(self):
        for actor in self.vector_actors:
            self.plotter.remove_actor(actor)
        self.vector_actors.clear()

    def clear_all(self):
        self.clear_trajectories()
        self.clear_markers()
        self.clear_vectors()
        self.clear_spacecraft()

    def show(self):
        self.plotter.show()


if __name__ == "__main__":
    canvas = Canvas(earth=Earth())
    canvas.add_spacecraft(Spacecraft(name="Vehicle", position=[7000.0, 0.0, 0.0], body_color="white", panel_color="silver"))
    canvas.show()