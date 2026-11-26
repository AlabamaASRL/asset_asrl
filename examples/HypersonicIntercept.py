# -*- coding: utf-8 -*-
"""
Minimum-Time Hypersonic Glide Vehicle Intercept

Models a powered interceptor engaging an unpowered hypersonic glide vehicle
(HGV) in a vertical plane.
"""

import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

#%% ASSET Imports

vf = ast.VectorFunctions
oc = ast.OptimalControl
ODEArgs = oc.ODEArguments
Args = vf.Arguments

#%% University of Alabama Plot Styling

ALABAMA_CRIMSON = "#9E1B32"
LIGHT_CRIMSON = "#C64B62"
PALE_CRIMSON = "#E18A9A"
LIGHT_GRAY = "#E6E6E6"
MUTED_GRAY = "#A8A8A8"
BACKGROUND = "#111111"
PANEL_BACKGROUND = "#1B1B1B"
GRID_COLOR = "#555555"
TEXT_COLOR = "#F2F2F2"

CASE_COLORS = [ALABAMA_CRIMSON, LIGHT_CRIMSON, PALE_CRIMSON]
HGV_COLOR = LIGHT_GRAY
GROUND_COLOR = MUTED_GRAY

FIGURE_DPI = 300
FIGURE_SIZE = (15, 10)

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": PANEL_BACKGROUND,
    "axes.edgecolor": MUTED_GRAY,
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": PANEL_BACKGROUND,
    "legend.edgecolor": MUTED_GRAY,
    "grid.color": GRID_COLOR,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "savefig.facecolor": BACKGROUND,
    "savefig.dpi": FIGURE_DPI
})


def style_axis(ax):
    ax.grid(True, alpha=0.4, linewidth=0.7)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_GRAY)
    ax.set_axisbelow(True)


def style_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    if legend:
        legend.get_frame().set_alpha(0.95)
        for item in legend.get_texts():
            item.set_color(TEXT_COLOR)
    return legend

#%% Constants
G0 = 9.80665
RHO0 = 1.225
ATMOSPHERE_SCALE_HEIGHT = 7200.0


def atmospheric_density(altitude):
    """Return exponential-atmosphere density in kg/m^3."""
    return RHO0 * vf.exp(-altitude / ATMOSPHERE_SCALE_HEIGHT)


def atmospheric_density_numpy(altitude):
    """Return exponential-atmosphere density using NumPy."""
    return RHO0 * np.exp(-max(float(altitude), 0.0) / ATMOSPHERE_SCALE_HEIGHT)

#%% Interceptor Configuration
T_MAX = 80.0e3
ISP = 250.0

S_INTERCEPTOR = 0.5
CL_ALPHA_INTERCEPTOR = 2.0
CD0_INTERCEPTOR = 0.05
K_INTERCEPTOR = 0.5

X0_I = 0.0
H0_I = 0.0
V0_I = 1200.0
GAMMA0_I = np.deg2rad(80.0)
M0_I = 1200.0

V_MIN_I = 100.0
V_MAX_I = 6500.0
M_MIN_I = 300.0
ALPHA_MAX_I = 0.3

#%% HGV Configuration

M_T = 1000.0
S_T = 5.0
CL_ALPHA_T = 2.0
CD0_T = 0.05
K_T = 0.5

ALPHA_T0 = 0.0
AMP_T = 0.1
OMEGA_T = 0.3

X0_T = 20000.0
H0_T = 35000.0
V0_T = 4500.0
GAMMA0_T = np.deg2rad(-12.0)

#%% Initial Guess Configuration

T_GUESS = 35.0
N_GUESS = 100
N_SEGMENTS = 40

INTERCEPTOR_CLIMB_BUMP = 8000.0
INITIAL_GUESS_THROTTLE = 0.25

#%% HGV Reference Trajectory

def hgv_rhs(time, state):
    xT, hT, vT, gammaT = state

    alphaT = ALPHA_T0 + AMP_T * np.sin(OMEGA_T * time)

    rhoT = atmospheric_density_numpy(hT)
    qT = 0.5 * rhoT * vT**2

    CLT = CL_ALPHA_T * alphaT
    CDT = CD0_T + K_T * CLT**2

    DT = qT * S_T * CDT
    LT = qT * S_T * CLT

    xTdot = vT * np.cos(gammaT)
    hTdot = vT * np.sin(gammaT)
    vTdot = -DT / M_T - G0 * np.sin(gammaT)
    gammaTdot = LT / (M_T * vT) - G0 * np.cos(gammaT) / vT

    return np.array([xTdot, hTdot, vTdot, gammaTdot])

def propagate_hgv_reference(final_time, num_points=N_GUESS):
    time_grid = np.linspace(0.0, final_time, num_points)
    states = np.zeros((num_points, 4))
    states[0] = [X0_T, H0_T, V0_T, GAMMA0_T]

    for i in range(num_points - 1):
        t = time_grid[i]
        dt = time_grid[i + 1] - time_grid[i]
        y = states[i]

        k1 = hgv_rhs(t, y)
        k2 = hgv_rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = hgv_rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = hgv_rhs(t + dt, y + dt * k3)

        states[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[i + 1, 1] = max(states[i + 1, 1], 0.0)
        states[i + 1, 2] = max(states[i + 1, 2], 1.0)

    return time_grid, states

#%% ODE Setup

class InterceptorTargetODE(oc.ODEBase):
    """
    Nine-state coupled interceptor and HGV point-mass dynamics.

    State ordering:
        0: Interceptor horizontal position
        1: Interceptor altitude
        2: Interceptor velocity
        3: Interceptor flight-path angle
        4: Interceptor mass
        5: HGV horizontal position
        6: HGV altitude
        7: HGV velocity
        8: HGV flight-path angle

    Control ordering:
        0: Interceptor angle of attack
        1: Interceptor throttle
    """

    def __init__(self, Tmax, Isp, S, CLa, CD0, k):
        nx, nu = 9, 2
        args = ODEArgs(nx, nu)

        xI, hI, vI, gammaI, mI, xT, hT, vT, gammaT = args.XVec().tolist()
        alphaI, throttle = args.UVar(0), args.UVar(1)

        # Interceptor aerodynamics.
        rhoI = atmospheric_density(hI)
        qI = 0.5 * rhoI * vI**2

        CLI = CLa * alphaI
        CDI = CD0 + k * CLI**2

        DI = qI * S * CDI
        LI = qI * S * CLI

        TI = Tmax * throttle

        # Interceptor equations of motion.
        xIdot = vI * vf.cos(gammaI)
        hIdot = vI * vf.sin(gammaI)
        vIdot = (TI * vf.cos(alphaI) - DI) / mI - G0 * vf.sin(gammaI)
        gammaIdot = (TI * vf.sin(alphaI) + LI) / (mI * vI) - G0 * vf.cos(gammaI) / vI
        mIdot = -TI / (Isp * G0)

        # Prescribed HGV angle of attack.
        time = args.TVar()
        alphaT = ALPHA_T0 + AMP_T * vf.sin(OMEGA_T * time)

        # HGV aerodynamics.
        rhoT = atmospheric_density(hT)
        qT = 0.5 * rhoT * vT**2

        CLT = CL_ALPHA_T * alphaT
        CDT = CD0_T + K_T * CLT**2

        DT = qT * S_T * CDT
        LT = qT * S_T * CLT

        # HGV equations of motion.
        xTdot = vT * vf.cos(gammaT)
        hTdot = vT * vf.sin(gammaT)
        vTdot = -DT / M_T - G0 * vf.sin(gammaT)
        gammaTdot = LT / (M_T * vT) - G0 * vf.cos(gammaT) / vT

        ode = vf.stack([xIdot, hIdot, vIdot, gammaIdot, mIdot, xTdot, hTdot, vTdot, gammaTdot])

        super().__init__(ode, nx, nu)

#%% Initial Guess

def build_initial_guess(final_fpa_deg, t_guess, seed=None):
    time_grid = np.linspace(0.0, 1.0, N_GUESS)
    gamma_f = np.deg2rad(final_fpa_deg)

    if seed is not None:
        seed_time = seed[9]
        seed_tf = seed_time[-1]
        initial_states = []

        for s in time_grid:
            t = seed_tf * s
            state = np.zeros(12)

            for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
                state[i] = np.interp(t, seed_time, seed[i])

            state[9] = t
            initial_states.append(state)

        return initial_states

    target_time, target_states = propagate_hgv_reference(t_guess)

    xT_final = target_states[-1, 0]
    hT_final = target_states[-1, 1]
    vT_final = target_states[-1, 2]

    mass_flow_max = T_MAX / (ISP * G0)
    guessed_mass_final = M0_I - INITIAL_GUESS_THROTTLE * mass_flow_max * t_guess
    guessed_mass_final = max(guessed_mass_final, M_MIN_I + 50.0)

    initial_states = []

    for s in time_grid:
        state = np.zeros(12)

        state[0] = X0_I + (xT_final - X0_I) * s
        state[1] = H0_I + (hT_final - H0_I) * s + INTERCEPTOR_CLIMB_BUMP * np.sin(np.pi * s)
        state[2] = V0_I + (vT_final - V0_I) * s
        state[3] = GAMMA0_I + (gamma_f - GAMMA0_I) * s
        state[4] = M0_I + (guessed_mass_final - M0_I) * s

        target_t = t_guess * s

        state[5] = np.interp(target_t, target_time, target_states[:, 0])
        state[6] = np.interp(target_t, target_time, target_states[:, 1])
        state[7] = np.interp(target_t, target_time, target_states[:, 2])
        state[8] = np.interp(target_t, target_time, target_states[:, 3])

        state[9] = target_t
        state[10] = 0.0
        state[11] = INITIAL_GUESS_THROTTLE

        initial_states.append(state)

    return initial_states

#%% Solve with ASSET

def solve(final_fpa_deg, t_guess=T_GUESS, seed=None):
    """Solve the minimum-time interceptor-HGV engagement problem."""

    ode = InterceptorTargetODE(T_MAX, ISP, S_INTERCEPTOR, CL_ALPHA_INTERCEPTOR, CD0_INTERCEPTOR, K_INTERCEPTOR)
    gamma_f = np.deg2rad(final_fpa_deg)
    initial_states = build_initial_guess(final_fpa_deg, t_guess, seed)

    phase = ode.phase("LGL3", initial_states, N_SEGMENTS)
    phase.setAutoScaling(True)
    phase.optimizer.PrintLevel = 0

    # Initial conditions.
    phase.addBoundaryValue("First", range(0, 10), [X0_I, H0_I, V0_I, GAMMA0_I, M0_I, X0_T, H0_T, V0_T, GAMMA0_T, 0.0])

    # Terminal interceptor flight-path angle.
    phase.addBoundaryValue("Last", [3], [gamma_f])

    # Terminal position matching.
    intercept_args = Args(4).tolist()
    intercept_function = vf.stack([intercept_args[0] - intercept_args[1], intercept_args[2] - intercept_args[3]])
    phase.addEqualCon("Last", intercept_function, [0, 5, 1, 6])

    # Interceptor path constraints.
    phase.addLowerVarBound("Path", 1, 0.0)
    phase.addLUVarBound("Path", 2, V_MIN_I, V_MAX_I)
    phase.addLUVarBound("Path", 3, -np.pi / 2, np.pi / 2)
    phase.addLowerVarBound("Path", 4, M_MIN_I)

    # HGV altitude constraint.
    phase.addLowerVarBound("Path", 6, 0.0)

    # Interceptor control constraints.
    phase.addLUVarBound("Path", 10, -ALPHA_MAX_I, ALPHA_MAX_I)
    phase.addLUVarBound("Path", 11, 0.0, 1.0)

    phase.addDeltaTimeObjective(1.0)
    phase.optimize()

    return np.array(phase.returnTraj()).T

#%% Plot Results

def plot_results(results):
    """Generate a four-panel Alabama-themed engagement summary."""

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)

    ax_geo, ax_vel, ax_gamma, ax_controls = axes.flatten()

    for i, (gamma_deg, traj) in enumerate(results.items()):
        color = CASE_COLORS[i % len(CASE_COLORS)]
        time = traj[9]
        tf = time[-1]
        label = rf"$\gamma_f={gamma_deg:.0f}^\circ$ ($t_f={tf:.1f}$ s)"

        ax_geo.plot(traj[0] / 1e3, traj[1] / 1e3, color=color, lw=2.5, label=f"{gamma_deg:.0f}° interceptor")
        ax_geo.plot(traj[5] / 1e3, traj[6] / 1e3, color=HGV_COLOR, lw=1.8, ls="--", alpha=0.85, label=f"{gamma_deg:.0f}° HGV")
        ax_geo.scatter(traj[0][-1] / 1e3, traj[1][-1] / 1e3, color=color, marker="X", edgecolor=LIGHT_GRAY, linewidth=0.8, s=75, zorder=5)

        ax_vel.plot(time, traj[2], color=color, lw=2.3, label=f"{gamma_deg:.0f}° interceptor")
        ax_vel.plot(time, traj[7], color=HGV_COLOR, lw=1.8, ls="--", alpha=0.85, label=f"{gamma_deg:.0f}° HGV")

        ax_gamma.plot(time, np.rad2deg(traj[3]), color=color, lw=2.3, label=label)
        ax_controls.plot(time, np.rad2deg(traj[10]), color=color, lw=2.3, label=f"{gamma_deg:.0f}° AOA")

    ax_geo.scatter(X0_I / 1e3, H0_I / 1e3, color=ALABAMA_CRIMSON, edgecolor=LIGHT_GRAY, marker="o", s=70, zorder=6, label="Interceptor launch")
    ax_geo.scatter(X0_T / 1e3, H0_T / 1e3, color=LIGHT_GRAY, edgecolor=ALABAMA_CRIMSON, marker="*", s=130, zorder=6, label="HGV start")

    ax_geo.set_xlabel(r"$x$ (km)")
    ax_geo.set_ylabel(r"$h$ (km)")
    ax_geo.set_title("Engagement Geometry", pad=10)
    style_legend(ax_geo, fontsize=7, ncol=2, loc="best")

    ax_vel.set_xlabel(r"$t$ (s)")
    ax_vel.set_ylabel(r"$v$ (m/s)")
    ax_vel.set_title("Vehicle Velocity", pad=10)
    style_legend(ax_vel, fontsize=8, ncol=2, loc="best")

    ax_gamma.set_xlabel(r"$t$ (s)")
    ax_gamma.set_ylabel(r"$\gamma_I$ (deg)")
    ax_gamma.set_title("Interceptor Flight-Path Angle", pad=10)
    style_legend(ax_gamma, fontsize=8, loc="best")

    ax_controls.set_xlabel(r"$t$ (s)")
    ax_controls.set_ylabel(r"$\alpha_I$ (deg)")
    ax_controls.set_title("Interceptor Angle of Attack and Throttle", pad=10)

    throttle_axis = ax_controls.twinx()
    throttle_axis.set_facecolor("none")

    for i, (gamma_deg, traj) in enumerate(results.items()):
        color = CASE_COLORS[i % len(CASE_COLORS)]
        throttle_axis.plot(traj[9], traj[11], color=color, lw=1.7, ls="--", alpha=0.8, label=f"{gamma_deg:.0f}° throttle")

    throttle_axis.set_ylabel("Throttle", color=TEXT_COLOR)
    throttle_axis.set_ylim(0.0, 1.05)
    throttle_axis.tick_params(axis="y", colors=TEXT_COLOR)

    for spine in throttle_axis.spines.values():
        spine.set_color(MUTED_GRAY)

    style_legend(ax_controls, fontsize=8, loc="upper left")
    style_axis(ax_geo)
    style_axis(ax_vel)
    style_axis(ax_gamma)
    style_axis(ax_controls)

    fig.suptitle("Minimum-Time Hypersonic Interceptor Engagement", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.985)
    fig.tight_layout(rect=(0.025, 0.025, 0.975, 0.94), h_pad=2.0, w_pad=1.8)
    fig.savefig("HGV_Interceptor.png", dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.12, facecolor=BACKGROUND)
    plt.show()

def make_animation(results, nframes=150, fps=15):
    cases = list(results.items())

    if not cases:
        raise ValueError("No engagement trajectories were provided.")

    final_times = [traj[9][-1] for _, traj in cases]
    max_time = max(final_times)

    time_grid = np.linspace(0.0, max_time, nframes)

    animation_data = []
    all_x = []
    all_h = []

    for gamma_deg, traj in cases:
        trajectory_time = traj[9]
        tf = trajectory_time[-1]
        valid_time = time_grid <= tf

        xI = np.interp(time_grid[valid_time], trajectory_time, traj[0]) / 1e3
        hI = np.interp(time_grid[valid_time], trajectory_time, traj[1]) / 1e3
        xT = np.interp(time_grid[valid_time], trajectory_time, traj[5]) / 1e3
        hT = np.interp(time_grid[valid_time], trajectory_time, traj[6]) / 1e3

        throttle = np.interp(time_grid[valid_time], trajectory_time, traj[11])

        all_x.extend([xI, xT])
        all_h.extend([hI, hT])

        animation_data.append({
            "gamma": gamma_deg,
            "tf": tf,
            "valid_time": valid_time,
            "time": time_grid[valid_time],
            "xI": xI,
            "hI": hI,
            "xT": xT,
            "hT": hT,
            "throttle": throttle,
            "color": CASE_COLORS[len(animation_data) % len(CASE_COLORS)]})

    xmin = min(np.min(x) for x in all_x)
    xmax = max(np.max(x) for x in all_x)
    hmin = min(np.min(h) for h in all_h)
    hmax = max(np.max(h) for h in all_h)

    xpadding = max(2.0, 0.04 * (xmax - xmin))
    hpadding = max(2.0, 0.05 * (hmax - hmin))

    common_xlim = (xmin - xpadding, xmax + xpadding)
    common_ylim = (min(-1.0, hmin - hpadding), hmax + hpadding)

    fig, ax = plt.subplots(figsize=(12, 7))

    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL_BACKGROUND)

    ax.set_xlim(common_xlim)
    ax.set_ylim(common_ylim)

    ax.set_xlabel(r"$x$ (km)", fontsize=11)
    ax.set_ylabel(r"$h$ (km)", fontsize=11)

    ax.set_title(
        "Hypersonic Interceptor Engagements",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=15)

    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.8)
    ax.axhline(0, color=GROUND_COLOR, lw=2.5, label="Sea level")

    for spine in ax.spines.values():
        spine.set_color(MUTED_GRAY)

    ax.tick_params(colors=TEXT_COLOR, labelsize=9)

    animated_artists = []

    for case in animation_data:

        gamma_deg = case["gamma"]
        color = case["color"]

        ax.plot(
            case["xI"],
            case["hI"],
            color=color,
            alpha=0.18,
            lw=1.5)

        ax.plot(
            case["xT"],
            case["hT"],
            color=color,
            alpha=0.18,
            lw=1.5,
            ls="--")

        trailI, = ax.plot(
            [],
            [],
            color=color,
            lw=2.8,
            label=rf"Interceptor $\gamma_f={gamma_deg:.0f}^\circ$",
            animated=True)

        trailT, = ax.plot(
            [],
            [],
            color=color,
            lw=2.0,
            ls="--",
            alpha=0.9,
            label=rf"HGV $\gamma_f={gamma_deg:.0f}^\circ$",
            animated=True)

        pointI, = ax.plot(
            [],
            [],
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=LIGHT_GRAY,
            linestyle="None",
            animated=True)

        pointT, = ax.plot(
            [],
            [],
            marker="^",
            markersize=9,
            markerfacecolor=color,
            markeredgecolor=LIGHT_GRAY,
            linestyle="None",
            animated=True)

        case["artists"] = (trailI, trailT, pointI, pointT)

        animated_artists.extend(case["artists"])

    timer = ax.text(
        0.025,
        0.96,
        "",
        transform=ax.transAxes,
        color=TEXT_COLOR,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=BACKGROUND,
            edgecolor=ALABAMA_CRIMSON,
            alpha=0.95),
        animated=True)

    animated_artists.append(timer)

    legend = ax.legend(loc="best", fontsize=8, ncol=2)
    legend.get_frame().set_facecolor(PANEL_BACKGROUND)
    legend.get_frame().set_edgecolor(MUTED_GRAY)

    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    def update(frame):
        current_time = time_grid[frame]

        for case in animation_data:
            trailI, trailT, pointI, pointT = case["artists"]
            n = np.searchsorted(case["time"], current_time, side="right")

            if current_time >= case["tf"]:
                n = len(case["time"])

            if n > 0:
                trailI.set_data(
                    case["xI"][:n],
                    case["hI"][:n])

                trailT.set_data(
                    case["xT"][:n],
                    case["hT"][:n])

                pointI.set_data(
                    [case["xI"][n - 1]],
                    [case["hI"][n - 1]])

                pointT.set_data(
                    [case["xT"][n - 1]],
                    [case["hT"][n - 1]])

            else:

                trailI.set_data([], [])
                trailT.set_data([], [])
                pointI.set_data([], [])
                pointT.set_data([], [])

        completed = sum(current_time >= case["tf"] for case in animation_data)

        timer.set_text(
            f"Elapsed Time: {current_time:.2f} s\n"
            f"Completed Intercepts: {completed}/{len(animation_data)}")

        return animated_artists

    animation = FuncAnimation(
        fig,
        update,
        frames=nframes,
        interval=1000 / fps,
        blit=True,
        cache_frame_data=False)

    filename = "HGV_Interceptor.gif"

    animation.save(
        filename,
        writer=PillowWriter(fps=fps),
        dpi=90,
        savefig_kwargs={
            "facecolor": BACKGROUND,
            "edgecolor": "none",
            "pad_inches": 0.02})

    plt.close(fig)

if __name__ == "__main__":
    results = {}
    seed = None

    for gamma_deg in [-30, -40, -50]:
        traj = solve(gamma_deg, t_guess=T_GUESS, seed=seed)
        results[gamma_deg] = traj
        seed = traj
        tf = traj[9][-1]
        miss_x = traj[0][-1] - traj[5][-1]
        miss_h = traj[1][-1] - traj[6][-1]
        achieved_gamma = np.rad2deg(traj[3][-1])
        
    print('\nMaking plots ...')
    make_animation(results)
    plot_results(results)