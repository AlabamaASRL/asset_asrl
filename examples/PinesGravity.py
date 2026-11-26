# -*- coding: utf-8 -*-
"""
Example use of ASSET's Pines gravity kernel.
"""

from functools import lru_cache

import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
from pathlib import Path

vf = ast.VectorFunctions
oc = ast.OptimalControl
astro = ast.Astro

MU = 3.9860043543609598e5
RE = 6378.1363

COEFF_FILE = str(Path(__file__).resolve().parents[1] / "native" / "data" / "egm96.nor")

@lru_cache(maxsize=None)
def MakePines(degree):
    """
    Build each gravity model once; coefficient parsing can take time.
    """
    return astro.PinesGravity(COEFF_FILE, degree, MU, RE, "fully_normalized")

class PinesModel(oc.ODEBase):
    """
    Two-body state dynamics using Pines gravity.

    The Pines model evaluates gravity at a body-fixed position. This simple
    example holds that frame fixed and does not model Earth rotation.
    """

    def __init__(self, gravity):
        xvars = 6
        args = oc.ODEArguments(xvars, 0)
        position = args.head3()
        velocity = args.segment3(3)

        acceleration = gravity(position)  # Includes central gravity.
        ode = vf.stack([velocity, acceleration])
        super().__init__(ode, xvars)

def make_earth_surface():
    # If we had common plotting library, this would go in there
    longitude = np.linspace(0.0, 2.0 * np.pi, 40)
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, 20)
    longitude, latitude = np.meshgrid(longitude, latitude)

    x = RE * np.cos(latitude) * np.cos(longitude)
    y = RE * np.cos(latitude) * np.sin(longitude)
    z = RE * np.sin(latitude)
    return x, y, z

if __name__ == "__main__":
    position = np.array([4510.0, -3120.0, 5230.0])
    radius = np.linalg.norm(position)
    two_body = -MU * position / radius**3

    print("Degree   |a| [km/s^2]       |a - a_2body| [km/s^2]")
    for degree in [0, 2, 8, 20]:
        gravity = MakePines(degree)
        acceleration = np.asarray(gravity.acceleration(position))
        difference = np.linalg.norm(acceleration - two_body)

    # Show that the Pines VectorFunction can also provide derivatives.
    gravity8 = MakePines(8)
    acceleration, jacobian = gravity8.compute_jacobian(position)
    print("\nDegree-8 acceleration [km/s^2]:", np.asarray(acceleration))
    print("Acceleration Jacobian [1/s^2]:\n", np.asarray(jacobian))

    # Circular orbit initial state, inclined by 51.6 degrees.
    initial_radius = 7000.0
    initial_speed = np.sqrt(MU / initial_radius)
    inclination = np.deg2rad(51.6)

    initial_state = np.zeros(7)
    initial_state[0:3] = [initial_radius, 0.0, 0.0]
    initial_state[3:6] = [0.0, initial_speed * np.cos(inclination), initial_speed * np.sin(inclination)]

    period = 2.0 * np.pi * np.sqrt(initial_radius**3 / MU)
    final_time = 10.0 * period

    # Integrate point-mass and degree-8 gravity models.
    trajectories = {}
    for degree in [0, 8]:
        ode = PinesModel(MakePines(degree))
        integrator = ode.integrator(30.0)
        integrator.setAbsTol(1.0e-8)
        trajectories[degree] = np.asarray(integrator.integrate_dense(initial_state, final_time, 2000)).T

    # Degree-zero circular-orbit half-period check.
    ode0 = PinesModel(MakePines(0))
    integrator0 = ode0.integrator(30.0)
    integrator0.setAbsTol(1.0e-8)
    half_period_state = np.asarray(integrator0.integrate(initial_state, period / 2.0))
    half_period_error = np.linalg.norm(half_period_state[0:3] + initial_state[0:3])
    print(f"\nHalf-period position error [km]: {half_period_error:.3e}")

    # Difference between the two trajectories at matching output times.
    time_hours = trajectories[0][6] / 3600.0
    position_difference = np.linalg.norm(trajectories[8][0:3] - trajectories[0][0:3],axis=0,)

    # Plot orbits in 3D with Earth for scale, plus the trajectory separation.
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    orbit_ax = fig.add_subplot(1, 2, 1, projection="3d")
    difference_ax = fig.add_subplot(1, 2, 2)

    earth_x, earth_y, earth_z = make_earth_surface()
    orbit_ax.plot_surface(
        earth_x,
        earth_y,
        earth_z,
        color="steelblue",
        alpha=0.25,
        linewidth=0,
        shade=True)

    colors = {0: "tab:blue", 8: "tab:orange"}
    for degree in [0, 8]:
        trajectory = trajectories[degree]
        orbit_ax.plot(
            trajectory[0],
            trajectory[1],
            trajectory[2],
            color=colors[degree],
            linewidth=1.6,
            label=f"Degree {degree}")

    orbit_ax.scatter(
        [initial_state[0]],
        [initial_state[1]],
        [initial_state[2]],
        color="black",
        s=24,
        label="Start")

    limit = 1.12 * initial_radius
    orbit_ax.set(
        xlim=(-limit, limit),
        ylim=(-limit, limit),
        zlim=(-limit, limit),
        xlabel="X [km]",
        ylabel="Y [km]",
        zlabel="Z [km]",
        title="Orbit Propagation")
    
    orbit_ax.set_box_aspect((1, 1, 1))
    orbit_ax.view_init(elev=25, azim=35)
    orbit_ax.legend(loc="upper left")
    orbit_ax.grid(True, alpha=0.3)

    difference_ax.plot(
        time_hours,
        position_difference,
        color="tab:purple",
        linewidth=2)
    
    difference_ax.scatter(
        [time_hours[-1]],
        [position_difference[-1]],
        color="tab:purple",
        s=30,
        zorder=3)

    difference_ax.set(
        xlabel="Time [hours]",
        ylabel="Position Difference [km]",
        title="Degree-8 vs. Point-Mass Trajectory")
    
    difference_ax.grid(True, alpha=0.3)

    fig.suptitle("Pines Gravity Model: Effect of EGM96 Harmonics")
    plt.show()