# -*- coding: utf-8 -*-
"""
Earth visualization module.

This module defines the Earth class, which extends CelestialBody with
optional visual layers for clouds, atmospheric glow, the equator, and
the reference axis. Each layer is rendered independently using PyVista
and can be enabled or disabled during initialization.

"""

import numpy as np
import pyvista as pv

from CelestialBody import CelestialBody


class Earth(CelestialBody):
    """
    Represent Earth with optional surface and reference visualization layers.
    """

    def __init__(self, radius=6378.145, texture="bluemarble-2048.png", cloud_texture="clouds_2048.png", show_clouds=True, show_atmosphere=True, show_equator=True, show_reference_axis=True):
        """
        Initialize Earth and configure its optional visualization layers.

        Args:
            radius (float): Mean Earth radius in the chosen distance units.
                Defaults to 6378.145.
            texture (str, optional): Path to the Earth surface texture image.
                Defaults to "bluemarble-2048.png".
            cloud_texture (str, optional): Path to the cloud texture image.
                Defaults to "clouds_2048.png".
            show_clouds (bool): Whether to render the cloud layer.
            show_atmosphere (bool): Whether to render the atmospheric shell.
            show_equator (bool): Whether to render the equator line.
            show_reference_axis (bool): Whether to render the polar reference axis.
        """
        super().__init__(name="Earth", radius=radius, texture=texture)

        self.cloud_texture = cloud_texture

        self.show_clouds = show_clouds
        self.show_atmosphere = show_atmosphere
        self.show_equator = show_equator
        self.show_reference_axis = show_reference_axis

        self.cloud_actor = None
        self.atmosphere_actor = None
        self.equator_actor = None
        self.reference_axis_actor = None

    def add_to_canvas(self, canvas):
        """
        Render Earth and all enabled visualization layers on a canvas.

        The surface is added first, followed by the optional clouds,
        atmosphere, equator, and reference axis. Actors for the additional
        layers are stored as instance attributes when created.

        Args:
            canvas: Canvas object containing a PyVista plotter.
        """
        # Earth surface
        super().add_to_canvas(canvas)

        # Clouds
        if self.show_clouds:
            self._add_clouds(canvas)

        # Atmosphere
        if self.show_atmosphere:
            self._add_atmosphere(canvas)

        # Equator
        if self.show_equator:
            self._add_equator(canvas)

        # Reference axis
        if self.show_reference_axis:
            self._add_reference_axis(canvas)

    def _add_clouds(self, canvas):
        """
        Render a translucent cloud layer slightly above Earth's surface.

        The cloud texture is resolved using the inherited texture-path
        helper. If the image cannot be found, the cloud layer is skipped.

        Args:
            canvas: Canvas object containing a PyVista plotter.
        """
        cloud_mesh = self.create_textured_sphere(radius=self.radius * 1.002)
        texture_path = self.resolve_texture(self.cloud_texture)

        if texture_path is None:
            return

        texture = pv.read_texture(texture_path)
        self.cloud_actor = canvas.plotter.add_mesh(cloud_mesh, texture=texture, opacity=0.35, smooth_shading=True, name="Clouds")

    def _add_atmosphere(self, canvas):
        """
        Render a faint, light-blue spherical shell around Earth.

        The shell is scaled slightly larger than Earth's surface to create
        a simple visual representation of the atmosphere.

        Args:
            canvas: Canvas object containing a PyVista plotter.
        """
        atmosphere = pv.Sphere(radius=self.radius * 1.015, theta_resolution=180, phi_resolution=90)
        self.atmosphere_actor = canvas.plotter.add_mesh(atmosphere, color="lightskyblue", opacity=0.06, smooth_shading=True, name="Atmosphere")

    def _add_equator(self, canvas):
        """
        Render Earth's equator as a circular line in the XY plane.

        The line is placed slightly above the surface to reduce visual
        overlap with the textured sphere.

        Args:
            canvas: Canvas object containing a PyVista plotter.
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 720)
        radius = self.radius * 1.003
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)))
        polyline = pv.lines_from_points(points)
        self.equator_actor = canvas.plotter.add_mesh(polyline, color="white", line_width=2.0, name="Equator")

    def _add_reference_axis(self, canvas):
        """
        Render Earth's polar reference axis through the north and south poles.

        The axis is represented by a vertical line extending beyond Earth's
        surface to make the body's orientation easier to interpret.

        Args:
            canvas: Canvas object containing a PyVista plotter.
        """
        radius = self.radius * 1.08
        points = np.array([[0.0, 0.0, -radius], [0.0, 0.0, radius]])
        axis = pv.lines_from_points(points)
        self.reference_axis_actor = canvas.plotter.add_mesh(axis, color="gray", line_width=2.0, name="Earth Reference Axis")