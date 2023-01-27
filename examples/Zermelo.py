import asset_asrl as ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#Change a few of the matplotlib label sizes for ease of reading plots
params = {'xtick.labelsize': 12, 'ytick.labelsize' : 12,
          'axes.labelsize':15, 'legend.fontsize':11}
mpl.rcParams.update(params) 


################################################################################
## Setup
oc = ast.OptimalControl
vf = ast.VectorFunctions

phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes


################################################################################
## Constants
nSeg = 250
tol = 1e-12
nVecPlot = 20
vecPlotScale = 0.1


################################################################################
## System Dynamics
class Zermelo(oc.ODEBase):
    def __init__(self, vMax, wFunc):
        Xvars = 2
        Uvars = 1
        
        #we use vf.Arguments as opposed to 
        #oc.ODEArguments because of the time dependent model
        args = vf.Arguments(Xvars + 1 + Uvars)
        xyt = args.head_3()
        th = args[3]

        wx, wy = wFunc(xyt)

        xD = vMax * vf.cos(th) + wx
        yD = vMax * vf.sin(th) + wy

        ode = vf.Stack([xD, yD])

        super().__init__(ode, Xvars, Uvars)


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
    phase.Threads = 10  # Equal to number of physical cores

    phase.setTraj(trajG, nSeg)

    # 3. Enforce start and end points
    phase.addBoundaryValue("Front", [0, 1], A)
    phase.addBoundaryValue("Front", [2], [0.0])
    phase.addBoundaryValue("Back", [0, 1], B)

    phase.addLUVarBound("Path", 3, -np.pi, np.pi, 1)

    # 4. Add objective function
    phase.addDeltaTimeObjective(1.0)

    # 5. Optimize
    phase.optimizer.set_EContol(tol)
    phase.optimizer.set_KKTtol(tol)
    phase.solve_optimize()

    return phase.returnTraj()


################################################################################
## 2D Plotting
def colorScale(x, left=[48, 59, 194], right=[208, 35, 70]):
    return [int(round((x * right[i]) + ((1 - x) * left[i])))/(256) for i in range(3)]

def plot2DTrajList(tList):
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 8))
    for i, t in enumerate(tList):
        clr = colorScale(i / len(tList))
        axes[0].plot([X[0] for X in t], [X[1] for X in t],
                     color = [(clr[0]), (clr[1]), (clr[2])],
                     label = "Path "+str(i))
        
        axes[1].plot([X[2] for X in t], [X[3] for X in t],
                     color = [(clr[0]), (clr[1]), (clr[2])])
    axes[0].grid(True)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    axes[1].grid(True)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("$\\theta$ (rad)")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig("Plots/Zermelo/CompareWindModels.svg",
                dpi = 500)
    plt.show()

# -------------------------------------


def plot2DTrajListVF(tList, wFunc_num):
    fig, axes = plt.subplots(1, 2, figsize = (12, 8))

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
            
    QV = axes[0].quiver(xPlot, yPlot, uPlot, vPlot, label = "Wind", color = "blue")
    

    # Overlay trajectories and control
    for i, t in enumerate(tList):
        clr = colorScale(i / len(tList))
        axes[0].plot([X[0] for X in t], [X[1] for X in t],
                     color = [(clr[0]), (clr[1]), (clr[2])],
                     label = "Path "+str(i))
        
        axes[1].plot([X[2] for X in t], [X[3] for X in t],
                     color = [(clr[0]), (clr[1]), (clr[2])])
        
    axes[0].grid(True)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    axes[1].grid(True)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("$\\theta$ (rad)")
    axes[0].legend(ncol =2)
    plt.tight_layout()
    plt.savefig("Plots/Zermelo/CompareBoatSpeed.svg",
                dpi = 500)
    plt.show()


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

    plot2DTrajList([test1, test2, test3, test4])


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
