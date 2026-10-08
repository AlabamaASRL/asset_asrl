# -*- coding: utf-8 -*-

"""
SatelliteDemo.py

Earth-centered satellite orbit visualization using the existing:
    - Earth
    - CelestialBody
    - Spacecraft
    - Canvas

Units:
    Position: km
    Time: s
    Gravitational parameter: km^3/s^2

Controls:
    SPACE : Pause / resume
    R     : Reset simulation
    RIGHT : Increase simulation speed
    LEFT  : Decrease simulation speed
    F     : Follow satellite

This is a circular-orbit visualization demo, not an ASSET propagation.
"""

import numpy as np

from Earth import Earth
from Spacecraft import Spacecraft
from Canvas import Canvas


# ============================================================================
# Orbit configuration
# ============================================================================

MU_EARTH = 398600.4418       # km^3/s^2
EARTH_RADIUS = 6378.145      # km

ORBIT_ALTITUDE = 700.0       # km above Earth's surface
ORBIT_RADIUS = EARTH_RADIUS + ORBIT_ALTITUDE

# Circular-orbit angular rate and period.
ORBIT_RATE = np.sqrt(
    MU_EARTH / ORBIT_RADIUS**3
)

ORBIT_PERIOD = 2.0 * np.pi / ORBIT_RATE


def satellite_position(time):
    """
    Return satellite position in an Earth-centered inertial-style frame.

    The circular orbit lies in the XY plane.
    """

    theta = ORBIT_RATE * time

    return np.array([
        ORBIT_RADIUS * np.cos(theta),
        ORBIT_RADIUS * np.sin(theta),
        0.0,
    ])


def create_orbit_trajectory(
    samples=1000,
):
    """
    Generate a complete circular orbit for display.
    """

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        samples,
    )

    return np.column_stack((
        ORBIT_RADIUS * np.cos(theta),
        ORBIT_RADIUS * np.sin(theta),
        np.zeros_like(theta),
    ))


def main():

    print("=" * 64)
    print("ASSET SATELLITE VISUALIZATION DEMO")
    print("=" * 64)

    print(f"Earth radius:       {EARTH_RADIUS:.3f} km")
    print(f"Orbit altitude:     {ORBIT_ALTITUDE:.3f} km")
    print(f"Orbit radius:       {ORBIT_RADIUS:.3f} km")
    print(f"Orbital period:     {ORBIT_PERIOD / 60.0:.2f} minutes")
    print(f"Orbital speed:      "
          f"{np.sqrt(MU_EARTH / ORBIT_RADIUS):.3f} km/s")

    # ------------------------------------------------------------------------
    # Earth
    # ------------------------------------------------------------------------

    earth = Earth(
        radius=EARTH_RADIUS,
        texture="bluemarble-2048.png",
        cloud_texture="clouds_2048.png",
        show_clouds=True,
        show_atmosphere=True,
        show_equator=True,
        show_reference_axis=True,
    )

    # ------------------------------------------------------------------------
    # Canvas
    # ------------------------------------------------------------------------

    canvas = Canvas(
        earth=earth,
        title="ASSET | Earth Satellite Orbit",
        window_size=(1500, 950),
        show_stars=True,
        show_controls=True,
        show_sim_time=True,
        show_asset_logo=True,
        show_alabama_logo=True,
    )

    # ------------------------------------------------------------------------
    # Orbit trajectory
    # ------------------------------------------------------------------------

    trajectory = create_orbit_trajectory()

    canvas.plot_trajectory(
        trajectory,
        color="cyan",
        width=3.0,
        label="Satellite Orbit",
    )

    canvas.plot_startpoint(
        trajectory,
        color="lime",
        size=14.0,
        label="Orbit Start",
    )

    # ------------------------------------------------------------------------
    # Satellite
    # ------------------------------------------------------------------------

    initial_position = satellite_position(0.0)

    satellite = Spacecraft(
        name="LEO Satellite",
        position=initial_position,
        body_color="white",
        panel_color="silver",
    )

    canvas.add_spacecraft(satellite)

    # ------------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------------

    # Show the entire orbit initially.
    canvas.set_camera_position(
        position=[
            2.8 * ORBIT_RADIUS,
            -2.8 * ORBIT_RADIUS,
            1.5 * ORBIT_RADIUS,
        ],
        focal_point=[0.0, 0.0, 0.0],
        up=[0.0, 0.0, 1.0],
    )

    # ------------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------------

    def update_satellite(time):
        """
        Update the spacecraft position using the shared Canvas clock.
        """

        position = satellite_position(time)

        canvas.update_spacecraft(position)

    def reset_satellite():
        """
        Restore the spacecraft to its initial orbital position.
        """

        canvas.update_spacecraft(
            satellite_position(0.0)
        )

    canvas.reset_callback = reset_satellite

    # The Canvas simulation time advances in simulated seconds.
    # A 30x initial scale is useful for viewing the orbit.
    canvas.add_animation(
        update_function=update_satellite,
        interval=30,
        time_scale=30.0,
    )

    # Set initial state explicitly.
    update_satellite(0.0)

    print()
    print("Controls:")
    print("  SPACE  Pause / resume")
    print("  R      Reset")
    print("  RIGHT  Increase speed")
    print("  LEFT   Decrease speed")
    print("  F      Follow satellite")
    print()
    print("Close the visualization window to exit.")
    print("=" * 64)

    canvas.show()


if __name__ == "__main__":
    main()
