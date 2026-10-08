# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv


class Spacecraft:

    def __init__(
        self,
        name="Spacecraft",
        position=None,
        scale=None,
        body_color="white",
        panel_color="silver",
        show_panels=True,
    ):
        # ================================================================
        # Configuration
        # ================================================================

        self.name = name

        if position is None:
            position = np.zeros(3, dtype=float)

        self.position = np.asarray(
            position,
            dtype=float,
        )

        if self.position.shape != (3,):
            raise ValueError(
                "Spacecraft position must contain exactly three values."
            )

        self.scale = scale

        self.body_color = body_color
        self.panel_color = panel_color

        self.show_panels = show_panels

        # ================================================================
        # Visualization actors
        # ================================================================

        self.actor = None
        self.panel_actors = []

        # ================================================================
        # Canvas reference
        # ================================================================

        self.canvas = None

    # ====================================================================
    # Add to Canvas
    # ====================================================================

    def add_to_canvas(
        self,
        canvas,
        scale=None,
    ):
        """
        Add the spacecraft geometry to a Canvas.
        """

        self.canvas = canvas

        # ------------------------------------------------------------
        # Determine scale
        # ------------------------------------------------------------

        if scale is not None:
            self.scale = float(scale)

        elif self.scale is None:
            self.scale = (
                canvas._get_reference_radius()
                * 0.035
            )

        self.scale = float(self.scale)

        # ------------------------------------------------------------
        # Remove an existing spacecraft if necessary
        # ------------------------------------------------------------

        self.remove_from_canvas()

        # ------------------------------------------------------------
        # Create spacecraft body
        # ------------------------------------------------------------

        body_mesh = pv.Cube(
            center=self.position,
            x_length=self.scale,
            y_length=0.65 * self.scale,
            z_length=0.65 * self.scale,
        )

        self.actor = canvas.plotter.add_mesh(
            body_mesh,
            color=self.body_color,
            smooth_shading=True,
            name=self.name,
        )

        # ------------------------------------------------------------
        # Create solar panels
        # ------------------------------------------------------------

        if self.show_panels:
            self._add_solar_panels(canvas)

        return self.actor

    # ====================================================================
    # Solar panels
    # ====================================================================

    def _add_solar_panels(
        self,
        canvas,
    ):
        """
        Create and add the two solar panels.
        """

        scale = self.scale

        panel_offset = 0.85 * scale
        panel_width = 0.70 * scale
        panel_height = 0.25 * scale

        # ------------------------------------------------------------
        # Left panel
        # ------------------------------------------------------------

        left_center = (
            self.position
            + np.array(
                [
                    -panel_offset,
                    0.0,
                    0.0,
                ]
            )
        )

        left_panel = pv.Cube(
            center=left_center,
            x_length=panel_width,
            y_length=panel_height,
            z_length=0.10 * scale,
        )

        left_actor = canvas.plotter.add_mesh(
            left_panel,
            color=self.panel_color,
            smooth_shading=True,
            name=f"{self.name} Left Panel",
        )

        # ------------------------------------------------------------
        # Right panel
        # ------------------------------------------------------------

        right_center = (
            self.position
            + np.array(
                [
                    panel_offset,
                    0.0,
                    0.0,
                ]
            )
        )

        right_panel = pv.Cube(
            center=right_center,
            x_length=panel_width,
            y_length=panel_height,
            z_length=0.10 * scale,
        )

        right_actor = canvas.plotter.add_mesh(
            right_panel,
            color=self.panel_color,
            smooth_shading=True,
            name=f"{self.name} Right Panel",
        )

        self.panel_actors = [
            left_actor,
            right_actor,
        ]

    # ====================================================================
    # Position
    # ====================================================================

    def set_position(
        self,
        position,
    ):
        """
        Set the spacecraft position without requiring
        it to already be displayed.
        """

        position = np.asarray(
            position,
            dtype=float,
        )

        if position.shape != (3,):
            raise ValueError(
                "Spacecraft position must contain exactly three values."
            )

        self.position = position.copy()

    def update_position(
        self,
        position,
    ):
        """
        Move the spacecraft and all of its visualization actors.
        """

        position = np.asarray(
            position,
            dtype=float,
        )

        if position.shape != (3,):
            raise ValueError(
                "Spacecraft position must contain exactly three values."
            )

        # ------------------------------------------------------------
        # If the spacecraft has not been added to a Canvas,
        # simply update its stored position.
        # ------------------------------------------------------------

        if self.actor is None:

            self.position = position.copy()

            return

        # ------------------------------------------------------------
        # Determine translation
        # ------------------------------------------------------------

        body_mesh = self._get_actor_mesh(
            self.actor
        )

        old_body_center = np.mean(
            body_mesh.points,
            axis=0,
        )

        delta = (
            position
            - old_body_center
        )

        # ------------------------------------------------------------
        # Move body
        # ------------------------------------------------------------

        self._translate_mesh(
            body_mesh,
            delta,
        )

        # ------------------------------------------------------------
        # Move solar panels
        # ------------------------------------------------------------

        for actor in self.panel_actors:

            panel_mesh = self._get_actor_mesh(
                actor
            )

            self._translate_mesh(
                panel_mesh,
                delta,
            )

        # ------------------------------------------------------------
        # Update stored position
        # ------------------------------------------------------------

        self.position = position.copy()

    # ====================================================================
    # Mesh helpers
    # ====================================================================

    @staticmethod
    def _get_actor_mesh(
        actor,
    ):
        """
        Return the PolyData associated with a PyVista actor.
        """

        try:
            return actor.mapper.dataset

        except Exception:
            return actor

    @staticmethod
    def _translate_mesh(
        mesh,
        delta,
    ):
        """
        Translate a mesh by delta.
        """

        mesh.points = (
            mesh.points + delta
        )

        try:
            mesh.GetPoints().Modified()

        except Exception:
            pass

        try:
            mesh.Modified()

        except Exception:
            pass

    # ====================================================================
    # Canvas
    # ====================================================================

    def remove_from_canvas(self):
        """
        Remove the spacecraft actors from the Canvas.
        """

        if self.canvas is None:
            return

        # ------------------------------------------------------------
        # Main spacecraft body
        # ------------------------------------------------------------

        if self.actor is not None:

            try:
                self.canvas.plotter.remove_actor(
                    self.actor
                )

            except Exception:
                pass

        # ------------------------------------------------------------
        # Solar panels
        # ------------------------------------------------------------

        for actor in self.panel_actors:

            try:
                self.canvas.plotter.remove_actor(
                    actor
                )

            except Exception:
                pass

        self.actor = None
        self.panel_actors = []

    # ====================================================================
    # Visibility
    # ====================================================================

    def show(self):

        if self.actor is not None:
            self.actor.SetVisibility(True)

        for actor in self.panel_actors:
            actor.SetVisibility(True)

    def hide(self):

        if self.actor is not None:
            self.actor.SetVisibility(False)

        for actor in self.panel_actors:
            actor.SetVisibility(False)

    # ====================================================================
    # Styling
    # ====================================================================

    def set_body_color(
        self,
        color,
    ):

        self.body_color = color

        if self.actor is not None:

            try:
                self.actor.prop.color = color

            except Exception:
                pass

    def set_panel_color(
        self,
        color,
    ):

        self.panel_color = color

        for actor in self.panel_actors:

            try:
                actor.prop.color = color

            except Exception:
                pass

    # ====================================================================
    # Scale
    # ====================================================================

    def set_scale(
        self,
        scale,
    ):
        """
        Change the spacecraft display scale.

        The spacecraft is rebuilt on the current Canvas.
        """

        scale = float(scale)

        if scale <= 0.0:
            raise ValueError(
                "Spacecraft scale must be positive."
            )

        self.scale = scale

        if self.canvas is None:
            return

        canvas = self.canvas

        self.add_to_canvas(
            canvas,
            scale=scale,
        )

    # ====================================================================
    # Information
    # ====================================================================

    def get_position(self):

        return self.position.copy()

    def get_scale(self):

        return self.scale

    # ====================================================================
    # String representation
    # ====================================================================

    def __repr__(self):

        return (
            f"Spacecraft("
            f"name={self.name!r}, "
            f"position={self.position.tolist()}, "
            f"scale={self.scale})"
        )