import asset as ast
import numpy as np
import copy
import plotly as py
import plotly.graph_objects as go

vf = ast.VectorFunctions
oc = ast.OptimalControl

TModes = oc.TranscriptionModes
PhaseRegs = oc.PhaseRegionFlags

# ------------------------------------------------------------------------------
## System Parameters
mu = 3.986e14  # Earth gravitational parameter
rho0 = 1.2  # Density at sea level
H = 7500  # Nominal atmosphere altitude
m = 750 / 2.2046226  # Vehicle mass
R = 6387000.0  # Radius of Earth
Aref = np.pi * (3 * 0.0254 / 2) ** 2  # Vehicle reference area


# ------------------------------------------------------------------------------
## System Dynamics
def ODE(A):
    irows = 6  # 4 state, 1 time, 1 control

    def rho(h):
        return rho0 * vf.exp(-h / H)

    def Cl(alpha):
        return 1.5658 * alpha

    def Cd(alpha):
        return 1.6537 * alpha * alpha + 0.0612

    def D(h, v, alpha):
        return 0.5 * rho(h) * v * v * Cd(alpha) * A

    def L(h, v, alpha):
        return 0.5 * rho(h) * v * v * Cl(alpha) * A

    args = vf.Arguments(irows)
    hh = args[0]  # Altitude
    th = args[1]  # Angle of position
    vv = args[2]  # Velocity magnitude
    g = args[3]  # Velocity orientation angle
    t = args[4]  # Time
    a = args[5]  # Angle of Attack

    h = hh * H
    r = h + [R]

    v = vv * H

    hD = v * vf.sin(g) / H
    thD = (v * vf.cos(g) / r) / (np.pi / 180)
    vD = -D(h, v, a) / m - mu * vf.sin(g) / r.squared() / H
    gD = L(h, v, a) / (m * v) + (
        v * r.inverse() - mu * (v * r.squared())
    ).inverse() * vf.cos(g)

    return vf.Stack([hD, thD, vD, gD])


# ------------------------------------------------------------------------------
## Objective
class maxKE(oc.StateObjective):
    def __init__(self):
        args = vf.Arguments(1)
        obj = -args[0] * args[0]
        super().__init__(obj, PhaseRegs.Back, [2])


# ------------------------------------------------------------------------------
## Main
def main():
    h0 = 80000 / H
    v0 = 5000 / H
    thf = 4  # * np.pi / 180
    hff = 0.001
    A0 = np.pi * (24 * 0.0254 / 2) ** 2

    dt = 0.4
    tf = 500
    nSeg = 350
    tol = 1e-10
    steps = 1000
    nTraj = 50

    ocp = oc.OptimalControlProblem()
    ode = oc.ode_x_x.ode(ODE(A0), 4, 1)

    def continueOn(hf, Achar, traj_hf):
        ode_local = oc.ode_x_x.ode(ODE(Achar), 4, 1)

        phase = ode_local.phase(TModes.LGL5)
        phase.Threads = 8

        phase.addBoundaryValue(PhaseRegs.Front, [0, 1, 2, 4], [h0, 0.0, v0, 0.0])
        phase.addBoundaryValue(PhaseRegs.Back, [0, 1], [hf, thf])

        phase.addLUVarBound(PhaseRegs.Back, 2, 0, 999, 1)

        phase.setControlMode(oc.BlockConstant)

        phase.addStateObjective(maxKE())

        phase.optimizer.EContol = tol
        phase.optimizer.KKTtol = tol
        phase.optimizer.PrintLevel = 2

        phase.setTraj(traj_hf, nSeg)

        phase.solve_optimize()

        # breakpoint()

        return phase.returnTraj()

    # breakpoint()

    itg = ode.integrator(dt)
    ig = np.array([h0, 0, v0, 0, 0, 0])
    TrajG = itg.integrate_dense(ig, tf, steps)

    breakpoint()
    hfVals = np.linspace(h0, hff, num=nTraj)
    traj_hf = [continueOn(hfVals[0], A0, TrajG)]
    # breakpoint()
    for hf in hfVals:
        traj_hf.append(continueOn(hf, A0, traj_hf[-1]))
        print(hf)
        # breakpoint()

    # breakpoint()

    # print("Incrementing effective area")
    # traj_A = [traj_hf[-1]]
    # AVals = np.linspace(A0, Aref, num=2 * nTraj)
    # for Ai in AVals:
    #     traj_A.append(continueOn(hff, Ai, traj_A[-1]))
    #     print(Ai)

    breakpoint()

    def plotTrajList(tList, name):
        data = []
        layout = go.Layout(showlegend=True)
        for t in tList:
            data.append(
                go.Scatter(x=[X[1] for X in t], y=[X[0] for X in t], mode="lines")
            )

        fig = go.Figure(data=data, layout=layout)
        # fig.update_layout(scene_aspectmode="data")

        fig.show()

    plotTrajList(traj_hf, "Continuation of Hypersonic Descent from Altitude to Zero")
    # plotTrajList(
    #     traj_A, "Continuation of Hypersonic Descent from Small Area to Larger Area"
    # )

    breakpoint()


# ------------------------------------------------------------------------------
## Run
if __name__ == "__main__":
    main()
