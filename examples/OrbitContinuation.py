import asset_asrl as ast
from asset_asrl.Astro.AstroModels import CR3BP
import numpy as np
import copy
import matplotlib.pyplot as plt


################################################################################
# Setup
oc = ast.OptimalControl
vf = ast.VectorFunctions

phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes

################################################################################
# Constants
mE = 5.9724e24  # Earth mass (kg)
mM = 0.07346e24  # Moon mass (kg)
lstar = 385000 #characteristic distance (km)

dt = 3.1415 / 10000

################################################################################
# System Dynamics
mu = mM / (mE + mM)
ode = CR3BP(mE, mM, lstar)

# Create integrator (= propagator)
odeItg = ode.integrator(dt)


################################################################################
# Solve for periodic orbit using initial guess ig
def solvePeriodic(ig, tf, ode, odeItg, fixInit=[0, 1, 2] ):
    # 1: Integrate initial guess
    steps = 1000
    trajGuess = odeItg.integrate_dense(ig, tf, steps)

    # 2: Create optimal control phase and assign guess
    odePhase = ode.phase("LGL3")  # LGL-3 collocation
    odePhase.Threads = 8  # Equal to number of physical cores

    nSeg = 150  # number of segments
    odePhase.setTraj(trajGuess, nSeg)
    for idx in fixInit:
        odePhase.addBoundaryValue("Front", [idx], [ig[idx]])
    odePhase.addBoundaryValue(
        "Front", [3, 6], [0.0, 0.0]  # Initial y, vx, t = 0
    )
    odePhase.addBoundaryValue(
        "Back", [1, 3, 5], [0.0, 0.0, 0.0]  # Final y, vx = 0
    )

    # 4: Solve
    tol = 1e-12
    odePhase.optimizer.set_EContol(tol)  # Equality constraint tolerance
    odePhase.solve()

    # 5: Get solution and return
    trajSol = odePhase.returnTraj()

    return trajSol

################################################################################
# Perform basic continuation of ig along x[cIdx] with step dx up to lim
def contin(ig, tf, cIdx, dx, lim, fixInit=[0, 1, 2]):
    trajList = []
    # Calculate the first orbit
    trajList.append(solvePeriodic(ig, tf, ode, odeItg, fixInit))
    sign = np.sign(trajList[-1][0][cIdx] - lim)
    signLast = sign
    while sign == signLast:
        # Our guess for this step is the result of the last step
        g = np.copy(trajList[-1][0])
        t = np.copy(trajList[-1][-1][6])
        print(g)

        # Increment the cIdx'th term
        g[cIdx] += dx
 
        # Pass to solvePeriodic
        sol = solvePeriodic(g, t, ode, odeItg, fixInit)

        # Save result
        trajList.append(copy.deepcopy(sol))

        # Check limit condition
        signLast = sign
        sign = np.sign(trajList[-1][0][cIdx] - lim)
    return trajList



################################################################################
# Use plotly to plot a list of trajectories
def plotTrajList(tList, proj = False):
    data = []
    if proj == False:
        fig, axes = plt.subplots(figsize = (8, 8))
        for t in tList:
            axes.plot([x[0] for x in t], [x[1] for x in t], color = "red")
        axes.grid(True)
        plt.tight_layout()
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        plt.tight_layout()
        plt.savefig("Plots/OrbitContinuation/Lyapunov.svg",
                dpi = 500)
        plt.show()
    elif proj == True:
        fig2=plt.figure(figsize=(8,8))
        axes = fig2.add_subplot(projection='3d')
        
        for t in tList[::5]:
            axes.plot3D([x[0] for x in t], [x[1] for x in t], [x[2] for x in t],
                        color = "blue")
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")
        
        plt.tight_layout()
        plt.savefig("Plots/OrbitContinuation/Halo.svg",
                dpi = 500)
        plt.show()
    
################################################################################
# Continuation - L1 Lyapunov
ig = np.zeros((7))
ig[0] = 0.8234  # Initial x
ig[4] = 0.1263  # Initial vy
tf = 1.3
tj = solvePeriodic(ig, tf, ode, odeItg)
tl = contin(tj[0], tj[-1][6], cIdx=0, dx=-0.001, lim=0.77)

tlp = []

for t in tl:
    tt = copy.deepcopy(t)
    t.reverse()
    t2 = [[x[0], -x[1], x[2]] for x in t]
    tlp.append(tt + t2)

plotTrajList(tlp)

################################################################################
# Continuation - Northern L1 Halo
ig = np.zeros((7))
ig[0] = 0.8234
ig[4] = 0.1263
tf = 1.3715
tj = solvePeriodic(ig, tf, ode, odeItg, fixInit=[1, 2, 5])
tl = contin(tj[0], tj[-1][6], cIdx=2, dx=0.001, lim=0.214, fixInit=[1, 2, 5])

tlp = []
for t in tl:
    tt = copy.deepcopy(t)
    t.reverse()
    t2 = [[x[0], -x[1], x[2]] for x in t]
    tlp.append(tt + t2)

plotTrajList(tlp, proj = True)