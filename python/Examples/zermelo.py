import asset as ast
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff


################################################################################
## Setup
oc = ast.OptimalControl
vf = ast.VectorFunctions

phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes


################################################################################
## Constants
nSeg = 150
tol = 1e-12
nVecPlot = 20
vecPlotScale = 0.1


################################################################################
## System Dynamics
class Zermelo(oc.ode_x_u.ode):
    def __init__(self, vMax, wFunc):
        XV = 2
        UV = 1

        args = vf.Arguments(XV + 1 + UV)
        xyt = args.head_3()
        th = args[3]

        wx, wy = wFunc(xyt)

        xD = vMax * vf.cos(th) + wx
        yD = vMax * vf.sin(th) + wy

        ode = vf.stack([xD, yD])

        super().__init__(ode, XV, UV)


################################################################################
## Wind Functions
def noWind(xyt):
    # No asset functions, just numbers
    return 0, 0


# -------------------------------------


def uniformWind(xyt, ang=135 * np.pi / 180, vel=2):
    # No asset functions, just numbers
    return vel * np.cos(ang), vel * np.sin(ang)


# -------------------------------------


def constantDirWind(xyt, ang=45 * np.pi / 180):
    vel = vf.cos(xyt.head2().norm())

    return vel * np.cos(ang), vel * np.sin(ang)


# -------------------------------------


def variableDirWind(xyt):
    vel = vf.sin(xyt.head2().norm())
    ang = 2 * (xyt[0] + xyt[1])

    return vel * vf.cos(ang), vel * vf.sin(ang)


################################################################################
## Solver function
def navigate(A, B, vM=1, wF=uniformWind):
    # 1. Initial guess is straight line
    dist = np.linalg.norm(B - A)
    t0 = dist / vM
    d = (B - A) / dist
    ang = np.arctan2(d[1], d[0])
    trajG = [
        np.array(list(A + d * x) + [t0 * x, ang]) for x in np.linspace(0, 1, num=nSeg)
    ]
    # 2. Initialize phase
    
    
    
    phase = Zermelo(vM, wF).phase(tModes.LGL3)

    ##Zermelo(vM, wF).vf().rpt(trajG[0],1000000)
    ##input("s")
    phase.Threads = 8

    phase.setTraj(trajG, nSeg)

    # 3. Enforce start and end points
    phase.addBoundaryValue(phaseRegs.Front, [0, 1], A)
    phase.addBoundaryValue(phaseRegs.Front, [2], [0.0])
    phase.addBoundaryValue(phaseRegs.Back, [0, 1], B)

    phase.addLUVarBound(phaseRegs.Path, 3, -np.pi, np.pi, 1)

    # 4. Add objective function
    phase.addDeltaTimeObjective(1.0)

    # 5. Optimize
    phase.optimizer.EContol = tol
    phase.optimizer.KKTtol = tol
    # phase.EnableVectorization = True
    phase.solve_optimize()

    return phase.returnTraj()


################################################################################
## 2D Plotting
def plot2DTrajList(tList, name):
    fig = make_subplots(rows=1, cols=2)
    for i, t in enumerate(tList):
        fig.add_trace(
            go.Scatter(
                x=[X[0] for X in t],
                y=[X[1] for X in t],
                mode="lines",
                name="Path {}".format(i),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[X[2] for X in t],
                y=[X[3] for X in t],
                mode="lines",
                name="Control {}".format(i),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(scene_aspectmode="data")

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Theta", row=1, col=2)

    fig.show()
    fig.write_html("./{}.html".format(name))


# -------------------------------------


def plot2DTrajListVF(tList, name, wFunc_num):
    fig = make_subplots(rows=1, cols=2)

    # Generate vector field for trajectory plot
    maxX = max([max([x[0] for x in t]) for t in tList])
    minX = min([min([x[0] for x in t]) for t in tList])
    maxY = max([max([x[1] for x in t]) for t in tList])
    minY = min([min([x[1] for x in t]) for t in tList])

    xRange = np.linspace(minX, maxX, num=nVecPlot)
    yRange = np.linspace(minY, maxY, num=nVecPlot)

    xPlot, yPlot = np.meshgrid(xRange, yRange)

    uPlot = np.zeros_like(xPlot)
    vPlot = np.zeros_like(xPlot)
    for i in range(nVecPlot):
        for j in range(nVecPlot):
            u_ij, v_ij = wFunc_num([xPlot[i, j], yPlot[i, j]])
            uPlot[i, j] = u_ij
            vPlot[i, j] = v_ij

    qv = ff.create_quiver(xPlot, yPlot, uPlot, vPlot)

    for d in qv.data:
        fig.add_trace(go.Scatter(x=d["x"], y=d["y"], name="Wind"), row=1, col=1)

    # Overlay trajectories and control
    for i, t in enumerate(tList):
        fig.add_trace(
            go.Scatter(
                x=[X[0] for X in t],
                y=[X[1] for X in t],
                mode="lines",
                name="Path {}".format(i),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[X[2] for X in t],
                y=[X[3] for X in t],
                mode="lines",
                name="Control {}".format(i),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(scene_aspectmode="data")

    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Theta", row=1, col=2)

    fig.show()
    fig.write_html("./{}.html".format(name))


################################################################################
## Compare Wind Models
def compareWind():
    A = np.array([0, -1])
    B = np.array([1, 1])
    vM = 1.25

    test1 = navigate(A, B, vM=1, wF=noWind)
    test2 = navigate(
        A,
        B,
        vM=vM,
        wF=lambda xyt: uniformWind(xyt, vel=0.5),
    )
    test3 = navigate(
        A,
        B,
        vM=vM,
        wF=constantDirWind,
    )
    test4 = navigate(A, B, vM=vM, wF=variableDirWind)

    plot2DTrajList([test1, test2, test3, test4], "CompareWindModels")


################################################################################
## Compare Boat Speed
def compareSpeed():
    A = np.array([-2, 2])
    B = np.array([1, 3])

    vMRange = np.linspace(1.1, 5, num=25)
    trajs = []
    for vM in vMRange:
        trajs.append(navigate(A, B, vM=vM, wF=variableDirWind))

    vdwx, vdwy = variableDirWind(vf.Arguments(2))

    plot2DTrajListVF(
        trajs,
        "CompareBoatSpeed",
        lambda xyt: (vdwx.compute(xyt), vdwy.compute(xyt)),
    )


################################################################################
## Main
def main():
    compareWind()
    compareSpeed()


################################################################################
## Run
if __name__ == "__main__":
    main()
