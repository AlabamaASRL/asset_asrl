# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:14:34 2026

@author: Sarah
"""

# CelestialBody.py

import os

import numpy as np
import pyvista as pv


class CelestialBody:

    def __init__(
        self,
        name,
        radius,
        texture=None,
    ):
        self.name = name
        self.radius = float(radius)
        self.texture = texture

        self.actor = None

    def resolve_texture(self, filename):

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

    def create_textured_sphere(
        self,
        radius=None,
        n_latitude=90,
        n_longitude=180,
    ):

        if radius is None:
            radius = self.radius

        longitude = np.linspace(
            0.0,
            2.0 * np.pi,
            n_longitude + 1,
        )

        latitude = np.linspace(
            -0.5 * np.pi,
            0.5 * np.pi,
            n_latitude + 1,
        )

        lon_grid, lat_grid = np.meshgrid(
            longitude,
            latitude,
        )

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

        points = np.column_stack(
            (
                x.ravel(),
                y.ravel(),
                z.ravel(),
            )
        )

        faces = []

        for i in range(n_latitude):

            for j in range(n_longitude):

                p0 = i * (n_longitude + 1) + j
                p1 = p0 + 1
                p2 = p1 + (n_longitude + 1)
                p3 = p0 + (n_longitude + 1)

                faces.extend(
                    [
                        4,
                        p0,
                        p1,
                        p2,
                        p3,
                    ]
                )

        faces = np.asarray(
            faces,
            dtype=np.int64,
        )

        sphere = pv.PolyData(
            points,
            faces,
        )

        u = lon_grid / (2.0 * np.pi)
        v = (lat_grid + 0.5 * np.pi) / np.pi

        sphere.active_texture_coordinates = (
            np.column_stack(
                (
                    u.ravel(),
                    v.ravel(),
                )
            )
        )

        return sphere

    def add_to_canvas(self, canvas):
        """Add this celestial body to a Canvas."""

        mesh = self.create_textured_sphere()

        texture_path = self.resolve_texture(
            self.texture
        )

        if texture_path is not None:

            texture = pv.read_texture(
                texture_path
            )

            self.actor = canvas.plotter.add_mesh(
                mesh,
                texture=texture,
                smooth_shading=True,
                name=self.name,
            )

        else:

            self.actor = canvas.plotter.add_mesh(
                mesh,
                color="white",
                smooth_shading=True,
                name=self.name,
            )

        return self.actor