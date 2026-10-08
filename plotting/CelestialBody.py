# -*- coding: utf-8 -*-
"""
Celestial body rendering utilities for PyVista.

This module defines the CelestialBody class, which creates a spherical mesh
with optional surface texturing and adds it to a PyVista-based visualization
canvas. Texture paths are resolved relative to the current working directory,
the module directory, or an explicitly supplied absolute path.
"""

import os

import numpy as np
import pyvista as pv


class CelestialBody:
    """
    Represent and render a spherical celestial body in a PyVista scene.
    """

    def __init__(self, name, radius, texture=None):
        """
        Initialize a celestial body with its name, radius, and optional texture.

        Args:
            name (str): Display name used to identify the body in the scene.
            radius (float): Radius of the body in the visualization's distance units.
            texture (str, optional): Path to an image used to texture the sphere.
                Defaults to None.
        """
        self.name = name
        self.radius = float(radius)
        self.texture = texture
        self.actor = None

    def resolve_texture(self, filename):
        """
        Find a texture image using absolute and relative path candidates.

        Args:
            filename (str or None): Texture path to resolve.

        Returns:
            str or None: Path to an existing texture file, or None if no file
            can be found.
        """
        if filename is None:
            return None

        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename

        candidates = [filename, os.path.join(os.path.dirname(__file__), filename), os.path.join(os.getcwd(), filename)]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return None

    def create_textured_sphere(self, radius=None, n_latitude=90, n_longitude=180):
        """
        Create a spherical mesh with texture coordinates.

        The sphere is generated from latitude and longitude grids. Texture
        coordinates map the mesh to an equirectangular image, with longitude
        spanning the horizontal direction and latitude spanning the vertical
        direction.

        Args:
            radius (float, optional): Radius of the generated sphere. Uses the
                body's configured radius when omitted.
            n_latitude (int): Number of latitude divisions. Defaults to 90.
            n_longitude (int): Number of longitude divisions. Defaults to 180.

        Returns:
            pyvista.PolyData: Spherical surface mesh with texture coordinates.
        """
        if radius is None:
            radius = self.radius

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

        sphere = pv.PolyData(points, np.asarray(faces, dtype=np.int64))

        u = lon_grid / (2.0 * np.pi)
        v = (lat_grid + 0.5 * np.pi) / np.pi
        sphere.active_texture_coordinates = np.column_stack((u.ravel(), v.ravel()))

        return sphere

    def add_to_canvas(self, canvas):
        """
        Add the celestial body to a canvas and return its PyVista actor.

        A textured sphere is rendered when the configured texture file is
        available. If the texture cannot be resolved, the body is rendered
        as a white sphere instead.

        Args:
            canvas: Canvas object containing a PyVista plotter.

        Returns:
            vtk.vtkActor: Actor representing the celestial body in the scene.
        """
        mesh = self.create_textured_sphere()
        texture_path = self.resolve_texture(self.texture)

        if texture_path is not None:
            texture = pv.read_texture(texture_path)
            self.actor = canvas.plotter.add_mesh(mesh, texture=texture, smooth_shading=True, name=self.name)
        else:
            self.actor = canvas.plotter.add_mesh(mesh, color="white", smooth_shading=True, name=self.name)

        return self.actor