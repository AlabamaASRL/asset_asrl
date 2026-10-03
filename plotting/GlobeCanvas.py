# -*- coding: utf-8 -*-

"""
GlobeCanvas.py

Reusable native VisPy 3D Earth visualization.

Features
--------
- Interactive 3D Earth
- High-resolution equirectangular Earth texture
- Optional atmosphere
- Equator
- Trajectory plotting
- Start/end markers
- Arbitrary vectors
- Camera framing
- Clear/reset functions
- Stand-alone test mode

Coordinate convention
---------------------
Public plotting functions accept Cartesian coordinates in km by default.

If normalized=True:

    normalized coordinate * Lstar / 1000 = km

Expected Earth texture
----------------------
The default texture is:

    world.200401.3x21600x10800.jpg

It should be an equirectangular 2:1 Earth map.

The NASA Blue Marble image can therefore be placed directly beside
this file.

Example
-------
from GlobeCanvas import GlobeCanvas

globe = GlobeCanvas(
    earth_radius_km=6378.145
)

globe.plot_trajectory(
    trajectory,
    normalized=True,
    Lstar=6378145
)

globe.show()
"""

import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np

from vispy import app, io, scene
from vispy.scene import visuals
from vispy.visuals.filters import TextureFilter


# ============================================================================
# Globe Canvas
# ============================================================================

class GlobeCanvas:
    """
    Persistent interactive 3D Earth visualization.
    """

    def __init__(
        self,
        earth_radius_km=6378.145,
        canvas_size=(1400, 900),
        background="black",
        texture_path=None,
        show_earth=True,
        show_atmosphere=True,
        show_equator=True,
        earth_subdivisions=160,
    ):

        self.earth_radius_km = float(earth_radius_km)
        self.earth_subdivisions = int(earth_subdivisions)

        # --------------------------------------------------------------------
        # Automatically use the NASA texture beside this file.
        # --------------------------------------------------------------------

        if texture_path is None:

            texture_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "blue_marble_earth.jpg",
            )

        self.texture_path = texture_path

        # --------------------------------------------------------------------
        # Canvas
        # --------------------------------------------------------------------

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            bgcolor=background,
            size=canvas_size,
            show=False,
        )

        self.view = self.canvas.central_widget.add_view()

        self.view.camera = scene.cameras.TurntableCamera(
            fov=45.0,
            azimuth=35.0,
            elevation=25.0,
            distance=self.earth_radius_km * 4.0,
        )

        # --------------------------------------------------------------------
        # Object containers
        # --------------------------------------------------------------------

        self.earth_objects = []
        self.trajectory_objects = []
        self.marker_objects = []
        self.vector_objects = []

        # --------------------------------------------------------------------
        # Earth
        # --------------------------------------------------------------------

        if show_earth:
            self._add_earth()

        # --------------------------------------------------------------------
        # Atmosphere
        # --------------------------------------------------------------------

        if show_atmosphere:
            self._add_atmosphere()

        # --------------------------------------------------------------------
        # Equator
        # --------------------------------------------------------------------

        if show_equator:
            self._add_equator()

    # ========================================================================
    # Earth
    # ========================================================================

    def _add_earth(self):
        if not os.path.isfile(self.texture_path):
            raise FileNotFoundError(
                "Earth texture was not found:\n\n"
                f"{self.texture_path}"
            )
    
        print(f"Loading Earth texture:\n  {self.texture_path}")
    
        image = Image.open(self.texture_path).convert("RGB")
    
        # Reduce the huge NASA texture for reasonable GPU usage.
        max_width = 8192
        if image.width > max_width:
            scale = max_width / image.width
            new_size = (
                int(image.width * scale),
                int(image.height * scale),
            )
            print(f"Resizing texture: {image.size} -> {new_size}")
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    
        self.earth_texture_image = np.asarray(image)
    
        print(
            "Earth texture loaded:"
            f" {self.earth_texture_image.shape[1]}"
            f" x {self.earth_texture_image.shape[0]}"
        )
    
        vertices, faces, texcoords = self._create_uv_sphere(
            radius=self.earth_radius_km,
            subdivisions=self.earth_subdivisions,
        )
    
        self.earth = scene.visuals.Mesh(
            vertices=vertices,
            faces=faces,
            color="white",
            shading="smooth",
            parent=self.view.scene,
        )
    
        self.earth_texture_filter = TextureFilter(
            texture=self.earth_texture_image,
            texcoords=texcoords,
        )
    
        self.earth.attach(self.earth_texture_filter)
    
        self.earth_objects.append(self.earth)

    # ========================================================================
    # UV Sphere
    # ========================================================================

    def _create_uv_sphere(self, radius, subdivisions):
        n_lat = int(subdivisions)
        n_lon = int(subdivisions * 2)
    
        lat = np.linspace(
            -0.5 * np.pi,
            0.5 * np.pi,
            n_lat + 1,
        )
    
        lon = np.linspace(
            -np.pi,
            np.pi,
            n_lon + 1,
        )
    
        lat_grid, lon_grid = np.meshgrid(
            lat,
            lon,
            indexing="ij",
        )
    
        # Geographic -> ECEF
        x = (
            radius
            * np.cos(lat_grid)
            * np.cos(lon_grid)
        )
    
        y = (
            radius
            * np.cos(lat_grid)
            * np.sin(lon_grid)
        )
    
        z = radius * np.sin(lat_grid)
    
        vertices = np.column_stack(
            (
                x.ravel(),
                y.ravel(),
                z.ravel(),
            )
        )
    
        # NASA Blue Marble is an equirectangular map:
        # left edge  = -180 deg longitude
        # right edge = +180 deg longitude
        # top        = +90 deg latitude
        # bottom     = -90 deg latitude
        u = (lon_grid + np.pi) / (2.0 * np.pi)
        v = (lat_grid + 0.5 * np.pi) / np.pi
    
        texcoords = np.column_stack(
            (
                u.ravel(),
                v.ravel(),
            )
        )
    
        faces = []
    
        for i in range(n_lat):
            for j in range(n_lon):
                a = i * (n_lon + 1) + j
                b = a + 1
                c = a + (n_lon + 1)
                d = c + 1
    
                faces.append((a, c, b))
                faces.append((b, c, d))
    
        faces = np.asarray(
            faces,
            dtype=np.uint32,
        )
    
        return vertices, faces, texcoords

    # ========================================================================
    # Atmosphere
    # ========================================================================

    def _add_atmosphere(self):

        self.atmosphere = visuals.Sphere(
            radius=self.earth_radius_km * 1.015,
            method="latitude",
            parent=self.view.scene,
            color=(0.15, 0.35, 0.65, 0.10),
            subdivisions=40,
        )

        self.earth_objects.append(
            self.atmosphere
        )

    # ========================================================================
    # Equator
    # ========================================================================

    def _add_equator(self):

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            500,
        )

        points = np.column_stack(
            (
                self.earth_radius_km * np.cos(theta),
                self.earth_radius_km * np.sin(theta),
                np.zeros_like(theta),
            )
        )

        self.equator = visuals.Line(
            pos=points,
            color=(0.5, 0.5, 0.5, 0.7),
            width=1.5,
            method="gl",
            parent=self.view.scene,
        )

        self.earth_objects.append(
            self.equator
        )

    # ========================================================================
    # Coordinate Conversion
    # ========================================================================

    def _convert_position(
        self,
        position,
        normalized=False,
        Lstar=None,
    ):

        position = np.asarray(
            position,
            dtype=float,
        )

        if normalized:

            if Lstar is None:

                raise ValueError(
                    "Lstar must be supplied for normalized coordinates."
                )

            position = position * Lstar / 1000.0

        return position

    # ========================================================================
    # Trajectory
    # ========================================================================

    def plot_trajectory(
        self,
        trajectory,
        color=(1.0, 1.0, 1.0, 1.0),
        width=3.0,
        normalized=False,
        Lstar=None,
    ):

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        if trajectory.ndim != 2:

            raise ValueError(
                "Trajectory must be a 2D array."
            )

        if trajectory.shape[1] < 3:

            raise ValueError(
                "Trajectory must contain at least three position columns."
            )

        xyz = self._convert_position(
            trajectory[:, 0:3],
            normalized=normalized,
            Lstar=Lstar,
        )

        line = visuals.Line(
            pos=xyz,
            color=color,
            width=width,
            method="gl",
            parent=self.view.scene,
        )

        self.trajectory_objects.append(
            line
        )

        return line

    # ========================================================================
    # Point
    # ========================================================================

    def plot_point(
        self,
        point,
        color=(1.0, 1.0, 1.0, 1.0),
        size=10.0,
        normalized=False,
        Lstar=None,
    ):

        point = self._convert_position(
            point,
            normalized=normalized,
            Lstar=Lstar,
        )

        point = np.asarray(
            point,
            dtype=float,
        ).reshape(1, 3)

        marker = visuals.Markers(
            parent=self.view.scene,
        )

        marker.set_data(
            point,
            face_color=color,
            edge_color=(1.0, 1.0, 1.0, 1.0),
            size=size,
        )

        self.marker_objects.append(
            marker
        )

        return marker

    # ========================================================================
    # Start Point
    # ========================================================================

    def plot_startpoint(
        self,
        trajectory,
        color=(1.0, 1.0, 1.0, 1.0),
        size=10.0,
        normalized=False,
        Lstar=None,
    ):

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        return self.plot_point(
            trajectory[0, 0:3],
            color=color,
            size=size,
            normalized=normalized,
            Lstar=Lstar,
        )

    # ========================================================================
    # End Point
    # ========================================================================

    def plot_endpoint(
        self,
        trajectory,
        color=(1.0, 1.0, 1.0, 1.0),
        size=10.0,
        normalized=False,
        Lstar=None,
    ):

        trajectory = np.asarray(
            trajectory,
            dtype=float,
        )

        return self.plot_point(
            trajectory[-1, 0:3],
            color=color,
            size=size,
            normalized=normalized,
            Lstar=Lstar,
        )

    # ========================================================================
    # Vector
    # ========================================================================

    def plot_vector(
        self,
        origin,
        vector,
        scale=1.0,
        color=(1.0, 1.0, 1.0, 1.0),
        width=2.0,
        normalized=False,
        Lstar=None,
    ):

        origin = self._convert_position(
            origin,
            normalized=normalized,
            Lstar=Lstar,
        )

        vector = np.asarray(
            vector,
            dtype=float,
        )

        if normalized:

            if Lstar is None:

                raise ValueError(
                    "Lstar must be supplied for normalized coordinates."
                )

            vector = vector * Lstar / 1000.0

        endpoint = origin + vector * scale

        points = np.vstack(
            (
                origin,
                endpoint,
            )
        )

        line = visuals.Line(
            pos=points,
            color=color,
            width=width,
            method="gl",
            parent=self.view.scene,
        )

        self.vector_objects.append(
            line
        )

        return line

    # ========================================================================
    # Camera Framing
    # ========================================================================

    def frame(
        self,
        padding=1.25,
    ):

        max_radius = self.earth_radius_km

        for line in self.trajectory_objects:

            try:

                pos = line.pos

            except Exception:

                continue

            if pos is not None and len(pos) > 0:

                radius = np.max(
                    np.linalg.norm(
                        pos,
                        axis=1,
                    )
                )

                max_radius = max(
                    max_radius,
                    radius,
                )

        self.view.camera.center = (
            0.0,
            0.0,
            0.0,
        )

        self.view.camera.distance = (
            max_radius * padding
        )

    # ========================================================================
    # Clear Trajectories
    # ========================================================================

    def clear_trajectories(self):

        for obj in self.trajectory_objects:

            try:

                obj.parent = None

            except Exception:

                pass

        self.trajectory_objects = []

    # ========================================================================
    # Clear Markers
    # ========================================================================

    def clear_markers(self):

        for obj in self.marker_objects:

            try:

                obj.parent = None

            except Exception:

                pass

        self.marker_objects = []

    # ========================================================================
    # Clear Vectors
    # ========================================================================

    def clear_vectors(self):

        for obj in self.vector_objects:

            try:

                obj.parent = None

            except Exception:

                pass

        self.vector_objects = []

    # ========================================================================
    # Clear All Plotted Data
    # ========================================================================

    def clear(self):

        self.clear_trajectories()
        self.clear_markers()
        self.clear_vectors()

    # ========================================================================
    # Show
    # ========================================================================

    def show(self):

        self.frame()

        self.canvas.show()

        app.run()


if __name__ == "__main__":
    print("Starting Earth + trajectory test...")

    globe = GlobeCanvas(
        earth_radius_km=6378.145,
        show_earth=True,
        show_atmosphere=True,
        show_equator=True,
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 1000)

    altitude_km = 500.0
    radius_km = 6378.145 + altitude_km

    trajectory = np.column_stack(
        (
            radius_km * np.cos(theta),
            radius_km * np.sin(theta),
            np.zeros_like(theta),
        )
    )

    globe.plot_trajectory(
        trajectory,
        color=(1.0, 0.2, 0.1, 1.0),
        width=4.0,
    )
    globe.show()
