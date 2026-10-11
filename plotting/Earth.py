# -*- coding: utf-8 -*-
"""
Earth visualization module.

Extends CelestialBody with optional visual layers for clouds, atmospheric
glow, the equator, and the polar reference axis.

Reference geometry is defined in Earth's local body-fixed coordinate frame:
    X-Y plane : Equatorial plane
    Z axis    : North/south polar axis

All geometry is scaled relative to Earth's radius and rendered independently
of the Earth surface texture.
"""

import numpy as np
import pyvista as pv

from CelestialBody import CelestialBody


class Earth(CelestialBody):
    """Represent Earth with optional surface and reference visualization layers."""

    def __init__(
        self,
        radius=6378.145,
        texture="bluemarble-2048.png",
        cloud_texture="clouds_2048.png",
        show_clouds=True,
        show_atmosphere=True,
        show_equator=True,
        show_reference_axis=True,
        equator_scale=1.003,
        reference_axis_scale=1.08,
        equator_color="white",
        reference_axis_color="gray",
        equator_width=2.0,
        reference_axis_width=2.0,
    ):
        """
        Initialize Earth and configure optional visualization layers.

        Args:
            radius (float): Earth mean radius in the chosen distance units.
            texture (str): Earth surface texture path.
            cloud_texture (str): Cloud texture path.
            show_clouds (bool): Enable the cloud layer.
            show_atmosphere (bool): Enable the atmospheric shell.
            show_equator (bool): Enable the equatorial reference circle.
            show_reference_axis (bool): Enable the polar reference axis.
            equator_scale (float): Equator radius relative to Earth radius.
            reference_axis_scale (float): Axis half-length relative to radius.
            equator_color (str): Equator line color.
            reference_axis_color (str): Polar axis line color.
            equator_width (float): Equator line width.
            reference_axis_width (float): Polar axis line width.
        """
        super().__init__(
            name="Earth",
            radius=radius,
            texture=texture,
        )

        if radius <= 0.0:
            raise ValueError("Earth radius must be positive.")

        if equator_scale <= 1.0:
            raise ValueError(
                "equator_scale must be greater than 1 to avoid "
                "overlapping the Earth surface."
            )

        if reference_axis_scale <= 1.0:
            raise ValueError(
                "reference_axis_scale must be greater than 1."
            )

        self.cloud_texture = cloud_texture

        self.show_clouds = show_clouds
        self.show_atmosphere = show_atmosphere
        self.show_equator = show_equator
        self.show_reference_axis = show_reference_axis

        self.equator_scale = equator_scale
        self.reference_axis_scale = reference_axis_scale

        self.equator_color = equator_color
        self.reference_axis_color = reference_axis_color

        self.equator_width = equator_width
        self.reference_axis_width = reference_axis_width

        self.cloud_actor = None
        self.atmosphere_actor = None
        self.equator_actor = None
        self.reference_axis_actor = None

    def add_to_canvas(self, canvas):
        """
        Render Earth and all enabled visualization layers.

        The surface is rendered first. Additional layers are independent
        actors, allowing their visibility to be changed without modifying
        the Earth surface mesh or texture.
        """
        super().add_to_canvas(canvas)

        if self.show_clouds:
            self._add_clouds(canvas)

        if self.show_atmosphere:
            self._add_atmosphere(canvas)

        if self.show_equator:
            self._add_equator(canvas)

        if self.show_reference_axis:
            self._add_reference_axis(canvas)

    def _add_clouds(self, canvas):
        """Render the translucent cloud layer."""
        texture_path = self.resolve_texture(self.cloud_texture)

        if texture_path is None:
            return

        cloud_mesh = self.create_textured_sphere(
            radius=self.radius * 1.002
        )

        texture = pv.read_texture(texture_path)

        self.cloud_actor = canvas.plotter.add_mesh(
            cloud_mesh,
            texture=texture,
            opacity=0.35,
            smooth_shading=True,
            name="Earth Clouds",
        )

    def _add_atmosphere(self, canvas):
        """Render a faint atmospheric shell."""
        atmosphere = pv.Sphere(
            radius=self.radius * 1.015,
            theta_resolution=180,
            phi_resolution=90,
        )

        self.atmosphere_actor = canvas.plotter.add_mesh(
            atmosphere,
            color="lightskyblue",
            opacity=0.06,
            smooth_shading=True,
            name="Earth Atmosphere",
        )

    def _add_equator(self, canvas):
        """
        Render Earth's equator in the body-fixed X-Y plane.

        The circle is offset slightly beyond Earth's nominal radius to
        reduce z-fighting with the textured surface. It is a separate
        line actor, so the texture remains unchanged.
        """
        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            721,
            endpoint=True,
        )

        equator_radius = self.radius * self.equator_scale

        points = np.column_stack(
            (
                equator_radius * np.cos(theta),
                equator_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )

        equator = pv.lines_from_points(points)

        self.equator_actor = canvas.plotter.add_mesh(
            equator,
            color=self.equator_color,
            line_width=self.equator_width,
            render_lines_as_tubes=False,
            name="Earth Equator",
        )

    def _add_reference_axis(self, canvas):
        """
        Render Earth's polar axis along the body-fixed Z axis.

        The line extends beyond both poles to clarify Earth's orientation.
        Its endpoints are determined by Earth's radius, keeping the axis
        correctly scaled when a different radius is supplied.
        """
        axis_extent = self.radius * self.reference_axis_scale

        points = np.array(
            [
                [0.0, 0.0, -axis_extent],
                [0.0, 0.0, axis_extent],
            ],
            dtype=float,
        )

        axis = pv.lines_from_points(points)

        self.reference_axis_actor = canvas.plotter.add_mesh(
            axis,
            color=self.reference_axis_color,
            line_width=self.reference_axis_width,
            render_lines_as_tubes=False,
            name="Earth Reference Axis",
        )

    def set_reference_geometry_visibility(
        self,
        equator=None,
        reference_axis=None,
    ):
        """
        Enable or disable reference geometry after it has been rendered.

        Args:
            equator (bool or None): Set equator visibility.
            reference_axis (bool or None): Set polar axis visibility.

        Note:
            The corresponding actors must already exist on the canvas.
            To create a previously disabled layer, enable it before calling
            add_to_canvas().
        """
        if equator is not None:
            self.show_equator = bool(equator)

            if self.equator_actor is not None:
                self.equator_actor.SetVisibility(
                    self.show_equator
                )

        if reference_axis is not None:
            self.show_reference_axis = bool(reference_axis)

            if self.reference_axis_actor is not None:
                self.reference_axis_actor.SetVisibility(
                    self.show_reference_axis
                )
