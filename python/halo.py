import asset as ast
import numpy as np
import plotly as py
import plotly.graph_objects as go
import copy


################################################################################
# Setup
oc = ast.OptimalControl
vf = ast.VectorFunctions

phaseRegs = oc.PhaseRegionFlags
tModes = oc.TranscriptionModes


################################################################################
# Constants
mE = 5.9724e24  # Earth mass
mM = 0.07346e24  # Moon mass

dt = 3.1415 / 10000


################################################################################
class CR3BP(oc.ode_6.ode):
    def __init__(self,mu):
        ############################################################
        args  = oc.ODEArguments(6)
        r = args.head3()
        v = args.segment3(3)
        
        x    = args[0]
        y    = args[1]
        xdot = args[3]
        ydot = args[4]
        
        p1loc = np.array([-mu,0,0])
        p2loc = np.array([1.0-mu,0,0])
        
        ##Gravity Terms (x,y,z)
        g1 = (r-p1loc).normalized_power3()*(mu-1.0)
        g2 = (r-p2loc).normalized_power3()*(-mu)
        
       
        rterms = vf.stack([2*ydot + x,
                           -2.0*xdot +y]).padded_lower(1)
        
    
        acc = vf.sum([g1,g2,rterms])
        ode = vf.stack([v,acc])
       
        ##############################################################
        super().__init__(ode,6)


###############################################################################
# System Dynamics
mu = mM / (mE + mM)
ode = CR3BP(mu)

# Create integrator (= propagator)
odeItg = ode.integrator(dt)


################################################################################
# Solve for periodic orbit using initial guess ig
def solvePeriodic(ig, tf, fixInit=[0, 1, 2]):
    # 1: Integrate initial guess
    steps = 1000
    trajGuess = odeItg.integrate_dense(ig, tf, steps)

    # 2: Create optimal control phase and assign guess
    odePhase = ode.phase(tModes.LGL3)  # LGL-3 collocation
    nSeg = 256  # number of segments
    odePhase.setTraj(trajGuess, nSeg)

    # 3: Set Boundary Constraints
    for idx in fixInit:
        odePhase.addBoundaryValue(phaseRegs.Front, [idx], [trajGuess[0][idx]])

    odePhase.addBoundaryValue(
        phaseRegs.Front, [1, 3, 5, 6], [0.0, 0.0, 0.0, 0.0]  # Initial y, vx, vz, t = 0
    )
    odePhase.addBoundaryValue(
        phaseRegs.Back, [1, 3, 5], [0.0, 0.0, 0.0]  # Final y, vx, vz = 0
    )

    # 4: Solve
    tol = 1e-12
    odePhase.optimizer.EContol = tol  # Equality constraint tolerance
    odePhase.solve()

    # 5: Get solution and return
    trajSol = odePhase.returnTraj()

    return trajSol


################################################################################
# Perform basic continuation of ig along x[cIdx] with step dx up to lim
def contin(ig, tf, cIdx, dx, lim, fixInit=[0, 1, 2]):
    trajList = []
    # Calculate the first orbit
    trajList.append(solvePeriodic(ig, tf, fixInit))
    sign = np.sign(trajList[-1][0][cIdx] - lim)
    signLast = sign
    while sign == signLast:
        # Our guess for this step is the result of the last step
        g = trajList[-1][0]
        t = trajList[-1][-1][6]

        # Increment the cIdx'th term
        g[cIdx] += dx

        # Pass to solvePeriodic
        sol = solvePeriodic(g, t, fixInit)

        # Save result
        trajList.append(copy.deepcopy(sol))

        # Check limit condition
        signLast = sign
        sign = np.sign(trajList[-1][0][cIdx] - lim)

    return trajList


################################################################################
# Use plotly to plot a list of trajectories
def plotTrajList(tList, name):
    data = []
    layout = go.Layout(showlegend=True)
    for t in tList:
        data.append(
            go.Scatter3d(
                x=[X[0] for X in t],
                y=[X[1] for X in t],
                z=[X[2] for X in t],
                mode="lines",
            )
        )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(scene_aspectmode="data")

    fig.show()
    fig.write_html("./{}.html".format(name))


################################################################################
# Continuation - L1 Lyapunov
ig = np.zeros((7))
ig[0] = 0.8234  # Initial x
ig[4] = 0.1263  # Initial vy
tf = 1.3
tj = solvePeriodic(ig, tf)
tl = contin(tj[0], tj[-1][6], cIdx=0, dx=-0.001, lim=0.77)

tlp = []
for t in tl:
    tt = copy.deepcopy(t)
    t.reverse()
    t2 = [[x[0], -x[1], x[2]] for x in t]
    tlp.append(tt + t2)

plotTrajList(tlp, "L1Lyapunov")


################################################################################
# Continuation - Northern L1 Halo
ig = np.zeros((7))
ig[0] = 0.8234
ig[4] = 0.1263
tf = 1.3715
tj = solvePeriodic(ig, tf, fixInit=[1, 2])
tl = contin(tj[0], tj[-1][6], cIdx=2, dx=0.001, lim=0.214, fixInit=[1, 2])

tlp = []
for t in tl:
    tt = copy.deepcopy(t)
    t.reverse()
    t2 = [[x[0], -x[1], x[2]] for x in t]
    tlp.append(tt + t2)

plotTrajList(tlp, "NorthL1Halo")
