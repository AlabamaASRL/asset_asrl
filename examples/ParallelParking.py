# -*- coding: utf-8 -*-
"""
Minimum-Time Parallel Parking Example Using ASSET

This example solves a minimum-time parallel parking problem using ASSET's
optimal control framework.

The problem is derived from the ICLOCS parallel parking example:

    http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html

The ICLOCS example is based on Case 7 in Table 2 of:

    https://ieeexplore.ieee.org/document/7463491

The initial conditions used here follow the referenced research paper.
The resulting maneuver time is compared with the reported value of
18.426 seconds.
"""

import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Polygon

vf = ast.VectorFunctions
oc = ast.OptimalControl
args = vf.Arguments


def corner_location(theta, loc_x, loc_y):
    """Rotate a body-frame point by the vehicle heading angle."""
    x_rotated = vf.cos(theta) * loc_x - vf.sin(theta) * loc_y
    y_rotated = vf.sin(theta) * loc_x + vf.cos(theta) * loc_y
    return x_rotated, y_rotated


def heaviside(x, k=10):
    """Smooth approximation of the Heaviside step function."""
    return (1.0 + vf.tanh(k * x)) / 2.0


def parking_slot_boundary(x, k, slot_length, slot_width):
    """Return the smoothed lower boundary of the parking space."""
    return (-heaviside(x, k) + heaviside(x - slot_length, k)) * slot_width


def triangle_area(point_a, point_b, point_c):
    """Calculate the area of a triangle using its vertex coordinates."""
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0


class CarODE(oc.ODEBase):
    """Kinematic bicycle model with vehicle geometry constraints."""

    def __init__(self, l_front, l_axes, l_rear, b_width):
        """Initialize the vehicle dimensions and equations of motion."""

        self.l_front = l_front
        self.l_axes = l_axes
        self.l_rear = l_rear
        self.b_width = b_width

        self.area_reference = (l_axes + l_front + l_rear) * 2.0 * b_width

        # Vehicle corners in the body frame, measured from the rear axle.
        self.corner_a = [l_front + l_axes, b_width]
        self.corner_b = [l_front + l_axes, -b_width]
        self.corner_c = [-l_rear, b_width]
        self.corner_d = [-l_rear, -b_width]
        self.corners = [self.corner_a, self.corner_b, self.corner_c, self.corner_d]

        state_control = oc.ODEArguments(6, 2)
        x, y, velocity, acceleration, theta, phi = state_control.XVec().tolist()
        jerk, steering_rate = state_control.UVec().tolist()

        x_dot = velocity * vf.cos(theta)
        y_dot = velocity * vf.sin(theta)
        velocity_dot = acceleration
        acceleration_dot = jerk
        theta_dot = velocity * vf.tan(phi) / self.l_axes
        phi_dot = steering_rate

        dynamics = vf.stack([x_dot, y_dot, velocity_dot, acceleration_dot, theta_dot, phi_dot])
        super().__init__(dynamics, 6, 2)

    def slot_bounds(self, slot_length, slot_width, curb_width):
        """Constrain all vehicle corners to remain within the roadway and parking-space boundaries."""

        x, y, theta, smoothing = args(4).tolist()
        constraints = []

        for loc_x, loc_y in self.corners:
            rotated_x, rotated_y = corner_location(theta, loc_x, loc_y)
            corner_x = x + rotated_x
            corner_y = y + rotated_y

            upper_boundary = corner_y - curb_width
            lower_boundary = -corner_y + parking_slot_boundary(corner_x, smoothing, slot_length, slot_width)

            constraints.extend([upper_boundary, lower_boundary])

        return vf.stack(constraints)

    def corner_collision_constraints(self, slot_length):
        """Prevent collisions with the two upper corners of the parking-space entrance."""

        x, y, theta = args(3).tolist()
        slot_corner_left = [0.0, 0.0]
        slot_corner_right = [slot_length, 0.0]

        vehicle_corners = []

        for loc_x, loc_y in self.corners:
            rotated_x, rotated_y = corner_location(theta, loc_x, loc_y)
            vehicle_corners.append([x + rotated_x, y + rotated_y])

        corner_a, corner_b, corner_c, corner_d = vehicle_corners

        left_areas = [
            triangle_area(slot_corner_left, corner_a, corner_b),
            triangle_area(slot_corner_left, corner_c, corner_b),
            triangle_area(slot_corner_left, corner_a, corner_d),
            triangle_area(slot_corner_left, corner_d, corner_c),
        ]

        right_areas = [
            triangle_area(slot_corner_right, corner_a, corner_b),
            triangle_area(slot_corner_right, corner_c, corner_b),
            triangle_area(slot_corner_right, corner_a, corner_d),
            triangle_area(slot_corner_right, corner_d, corner_c),
        ]

        constraint_left = self.area_reference - vf.sum(left_areas)
        constraint_right = self.area_reference - vf.sum(right_areas)

        return vf.stack(constraint_left, constraint_right)

    def final_y_constraints(self):
        """Constrain all four vehicle corners to lie within the parking-space width."""

        y, theta = args(2).tolist()
        constraints = []

        for loc_x, loc_y in self.corners:
            _, rotated_y = corner_location(theta, loc_x, loc_y)
            constraints.append(y + rotated_y)

        return vf.stack(constraints)

    def curvature_rate(self):
        """Return the rate of change of vehicle curvature."""

        phi, steering_rate = args(2).tolist()
        return steering_rate / (self.l_axes * vf.cos(phi) ** 2)


def make_state(x, y, theta_degrees, time):
    """Construct an ASSET state-control vector for the initial guess."""

    state = np.zeros(9)
    state[0] = x
    state[1] = y
    state[4] = np.deg2rad(theta_degrees)
    state[6] = time

    return state


def rotation_matrix(angle):
    """Return a two-dimensional rotation matrix."""
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def transform_point(position, heading, local_point):
    """Transform a body-frame point into the inertial frame."""
    return np.asarray(position) + rotation_matrix(heading) @ np.asarray(local_point)


def vehicle_body_points(vehicle, state):
    """Return the vehicle body polygon in inertial coordinates."""

    position = state[0:2]
    heading = state[4]
    local_points = [vehicle.corner_a, vehicle.corner_c, vehicle.corner_d, vehicle.corner_b]

    return np.array([transform_point(position, heading, point) for point in local_points])


def wheel_points(vehicle, state, axle_position, steering_angle=0.0):
    """Return the four corners of a wheel in inertial coordinates."""

    position = state[0:2]
    heading = state[4]
    wheel_length = 0.30
    wheel_width = 0.12

    local_points = np.array([
        [wheel_length, wheel_width],
        [-wheel_length, wheel_width],
        [-wheel_length, -wheel_width],
        [wheel_length, -wheel_width],
    ])

    rotated_points = (rotation_matrix(steering_angle) @ local_points.T).T
    wheel_center = np.asarray(axle_position)

    return np.array([transform_point(position, heading, wheel_center + point) for point in rotated_points])


def vehicle_wheel_geometry(vehicle, state):
    """Return the front and rear wheel polygons."""

    axle_width = vehicle.b_width / 1.5
    rear_centers = [[0.0, axle_width], [0.0, -axle_width]]
    front_centers = [[vehicle.l_axes, axle_width], [vehicle.l_axes, -axle_width]]

    rear_wheels = [wheel_points(vehicle, state, center) for center in rear_centers]
    front_wheels = [wheel_points(vehicle, state, center, state[5]) for center in front_centers]

    return rear_wheels, front_wheels


def vehicle_axle_geometry(vehicle, state):
    """Return the rear and front axle line segments."""

    position = state[0:2]
    heading = state[4]
    axle_width = vehicle.b_width / 1.5

    rear_axle = np.array([
        transform_point(position, heading, [0.0, axle_width]),
        transform_point(position, heading, [0.0, -axle_width]),
    ])

    front_axle = np.array([
        transform_point(position, heading, [vehicle.l_axes, axle_width]),
        transform_point(position, heading, [vehicle.l_axes, -axle_width]),
    ])

    return rear_axle, front_axle


def plot_slot(ax, smoothing, slot_length, slot_width, curb_width, x_limits=(-8.0, 8.0)):
    """Plot the roadway, parking space, and smoothed slot boundary."""

    x_values = np.linspace(x_limits[0], x_limits[1], 2000)
    step_left = (1.0 + np.tanh(smoothing * x_values)) / 2.0
    step_right = (1.0 + np.tanh(smoothing * (x_values - slot_length))) / 2.0
    y_values = (-step_left + step_right) * slot_width

    ax.plot(x_values, y_values, color="red", linewidth=1.5, label="Parking boundary")
    ax.fill_between(x_values, 0.0, curb_width, color="grey", alpha=0.2, label="Roadway")
    ax.fill_between(x_values, y_values, 0.0, color="green", alpha=0.2, label="Parking space")
    ax.plot(x_limits, [curb_width, curb_width], color="red", linewidth=1.5)


def plot_vehicle(vehicle, state, ax, color="blue", alpha=1.0):
    """Plot the vehicle body, wheels, and axles at one trajectory point."""

    body = vehicle_body_points(vehicle, state)
    ax.fill(body[:, 0], body[:, 1], facecolor=color, edgecolor="black", linewidth=1.0, alpha=alpha, label="Vehicle")

    rear_wheels, front_wheels = vehicle_wheel_geometry(vehicle, state)

    for wheel in rear_wheels + front_wheels:
        ax.fill(wheel[:, 0], wheel[:, 1], facecolor="black", edgecolor="black", alpha=alpha)

    rear_axle, front_axle = vehicle_axle_geometry(vehicle, state)

    for axle in [rear_axle, front_axle]:
        ax.plot(axle[:, 0], axle[:, 1], color="black", linewidth=1.0, alpha=alpha)


def plot_trajectory(vehicle, trajectory, smoothing, slot_length, slot_width, curb_width, x_limits=(-6.3, 10.0)):
    """Plot the roadway, parking space, and vehicle trajectory."""

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_slot(ax, smoothing, slot_length, slot_width, curb_width, x_limits)

    colors = plt.get_cmap("viridis")(np.linspace(0.0, 1.0, len(trajectory)))

    for state, color in zip(trajectory, colors):
        plot_vehicle(vehicle, state, ax, color=color, alpha=0.85)

    ax.set_xlabel(r"$X$ (m)")
    ax.set_ylabel(r"$Y$ (m)")
    ax.set_title("Parallel Parking Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def animate_trajectory(vehicle, trajectory, smoothing, slot_length, slot_width, curb_width, save=False, filename="parallel_parking.mp4"):
    """Animate the vehicle maneuver and its trajectory history."""

    if len(trajectory) < 2:
        raise ValueError("The trajectory must contain at least two states.")

    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    plot_slot(ax, smoothing, slot_length, slot_width, curb_width, x_limits=(-8.0, 10.0))

    body_patch = Polygon(np.zeros((4, 2)), closed=True, facecolor="royalblue", edgecolor="black", linewidth=1.5, label="Vehicle")
    ax.add_patch(body_patch)

    wheel_patches = []

    for _ in range(4):
        wheel_patch = Polygon(np.zeros((4, 2)), closed=True, facecolor="black", edgecolor="black")
        ax.add_patch(wheel_patch)
        wheel_patches.append(wheel_patch)

    rear_axle_line, = ax.plot([], [], color="black", linewidth=1.5)
    front_axle_line, = ax.plot([], [], color="black", linewidth=1.5)
    trajectory_line, = ax.plot([], [], color="blue", linestyle="dotted", linewidth=1.5, label="Trajectory")
    current_position, = ax.plot([], [], color="black", marker="o", markersize=5, linestyle="none")
    time_text = ax.text(0.03, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top")

    artists = [body_patch, *wheel_patches, rear_axle_line, front_axle_line, trajectory_line, current_position, time_text]

    def initialize():
        """Initialize all animation artists."""

        body_patch.set_xy(np.zeros((4, 2)))

        for wheel in wheel_patches:
            wheel.set_xy(np.zeros((4, 2)))

        rear_axle_line.set_data([], [])
        front_axle_line.set_data([], [])
        trajectory_line.set_data([], [])
        current_position.set_data([], [])
        time_text.set_text("")

        return artists

    def update(frame):
        """Update the vehicle geometry for one animation frame."""

        state = trajectory[frame]
        position = state[0:2]
        body = vehicle_body_points(vehicle, state)
        body_patch.set_xy(body)

        rear_wheels, front_wheels = vehicle_wheel_geometry(vehicle, state)

        for wheel_patch, wheel_geometry in zip(wheel_patches, rear_wheels + front_wheels):
            wheel_patch.set_xy(wheel_geometry)

        rear_axle, front_axle = vehicle_axle_geometry(vehicle, state)
        rear_axle_line.set_data(rear_axle[:, 0], rear_axle[:, 1])
        front_axle_line.set_data(front_axle[:, 0], front_axle[:, 1])

        trajectory_history = np.asarray([sample[0:2] for sample in trajectory[:frame + 1]])
        trajectory_line.set_data(trajectory_history[:, 0], trajectory_history[:, 1])
        current_position.set_data([position[0]], [position[1]])
        time_text.set_text(f"Time = {state[6]:.2f} s")

        return artists

    duration = trajectory[-1][6] - trajectory[0][6]
    interval = max(1, int(1000.0 * duration / (len(trajectory) - 1)))

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=True, init_func=initialize, repeat_delay=5000)

    ax.set_xlabel(r"$X$ (m)")
    ax.set_ylabel(r"$Y$ (m)")
    ax.set_title("Minimum-Time Parallel Parking Maneuver")
    ax.set_xlim([-6.1, 8.0])
    ax.set_ylim([-2.8, 4.2])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save:
        writer = animation.FFMpegWriter(fps=30)
        ani.save(filename, writer=writer, dpi=300)

    plt.show()

    return ani


def main():
    """Construct, solve, and visualize the parallel parking problem."""

    # Parking-space dimensions.
    slot_length = 5.0
    slot_width = 2.0
    curb_width = 3.5

    # Vehicle dimensions.
    l_front = 0.839
    l_axes = 2.588
    l_rear = 0.657
    b_width = 1.771 / 2.0

    vehicle = CarODE(l_front, l_axes, l_rear, b_width)

    # Vehicle and control limits.
    max_steering_angle = np.deg2rad(33.0)
    max_velocity = 2.0
    max_acceleration = 0.75
    max_jerk = 0.5
    max_curvature_rate = 0.6

    min_x = -10.0
    max_x = 7.5

    # Initial conditions from the reference paper.
    initial_x = -5.14
    initial_y = 1.41
    initial_heading_degrees = 13.18

    initial_state = make_state(initial_x, initial_y, initial_heading_degrees, 0.0)

    # Initial trajectory guess.
    guess_1 = make_state(0.0, initial_y, 0.0, 5.0)
    guess_2 = make_state(5.5, initial_y, 10.0, 10.0)
    guess_3 = make_state(1.0, -0.5, 20.0, 15.0)
    guess_4 = make_state(1.0, -1.0, 0.0, 25.0)

    trajectory_guess = [initial_state, guess_1, guess_2, guess_3, guess_4]

    # Display the initial trajectory guess.
    plot_trajectory(vehicle, trajectory_guess, 1000.0, slot_length, slot_width, curb_width)
    plt.show()

    # Optimization and mesh-refinement parameters.
    initial_segments = 50
    refined_segments = 200
    initial_smoothing = 75.0
    refined_smoothing = 150.0

    # Construct the optimal control phase.
    phase = vehicle.phase("LGL5", trajectory_guess, initial_segments, True)
    phase.setStaticParams([initial_smoothing])
    phase.setControlMode("BlockConstant")

    # Fix the initial vehicle state.
    phase.addBoundaryValue("First", range(0, 7), initial_state[0:7])

    # Enforce roadway and parking-space boundaries.
    phase.addInequalCon("Path", vehicle.slot_bounds(slot_length, slot_width, curb_width), [0, 1, 4], [], [0])

    # Require the final vehicle corners to lie within the slot.
    phase.addInequalCon("Last", vehicle.final_y_constraints(), [1, 4])

    # Require zero terminal velocity and acceleration.
    phase.addBoundaryValue("Last", [2, 3], [0.0, 0.0])

    # State and control bounds.
    phase.addLUVarBound("Path", 0, min_x, max_x)
    phase.addLUVarBound("Path", 2, -max_velocity, max_velocity)
    phase.addLUVarBound("Path", 3, -max_acceleration, max_acceleration)
    phase.addLUVarBound("Path", 5, -max_steering_angle, max_steering_angle)
    phase.addLUVarBound("Path", 7, -max_jerk, max_jerk)

    # Bound the rate of change of curvature.
    phase.addLUFuncBound("Path", vehicle.curvature_rate(), [5, 8], -max_curvature_rate, max_curvature_rate)

    # Prevent collisions with the parking-space entrance corners.
    phase.addInequalCon("Path", vehicle.corner_collision_constraints(slot_length), [0, 1, 4])

    # Keep the initial smoothing parameter fixed during the first solve.
    phase.addValueLock("StaticParams", [0])

    # First optimization.
    phase.addDeltaTimeObjective(1)
    phase.optimizer.set_BoundFraction(0.995)
    phase.optimizer.set_MaxIters(2000)
    phase.optimizer.set_PrintLevel(1)
    phase.solve_optimize()

    # Mesh refinement and second optimization.
    phase.refineTrajManual(refined_segments)
    phase.subVariable("StaticParams", 0, refined_smoothing)
    phase.optimizer.set_KKTtol(1.0e-8)
    phase.optimize()

    # Extract optimized trajectory.
    trajectory = phase.returnTraj()
    trajectory_table = phase.returnTrajTable()

    # Independently reintegrate the optimized trajectory.
    integrator = vehicle.integrator(0.1, trajectory_table)
    reintegrated_trajectory = integrator.integrate_dense(trajectory[0], trajectory[-1][6], 500)

    # Report maneuver time and terminal-state consistency.
    final_time = trajectory[-1][6]
    reference_time = 18.426

    print()
    print("=" * 60)
    print("PARALLEL PARKING OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"ASSET maneuver time:      {final_time:.6f} s")
    print(f"Reference maneuver time:  {reference_time:.6f} s")

    time_difference = final_time - reference_time
    time_difference_percent = 100.0 * time_difference / reference_time

    print(f"Time difference:          {time_difference:.6f} s")
    print(f"Relative time difference: {time_difference_percent:.4f} %")

    terminal_state_error = trajectory[-1][0:7] - reintegrated_trajectory[-1][0:7]

    print()
    print("Optimized versus reintegrated terminal-state difference:")
    print(terminal_state_error)
    print("Terminal-state difference norm:", np.linalg.norm(terminal_state_error))
    print("=" * 60)

    # Plot optimized trajectory.
    plot_trajectory(vehicle, trajectory, refined_smoothing, slot_length, slot_width, curb_width)
    plt.show()

    # Animate the reintegrated trajectory.
    animate_trajectory(vehicle, reintegrated_trajectory, refined_smoothing, slot_length, slot_width, curb_width, save=False)

if __name__ == "__main__":
    main()