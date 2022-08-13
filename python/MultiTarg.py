import numpy as np
import asset as ast
import time
import copy
from plotly.subplots import make_subplots
import plotly.graph_objects as go


################################################################################
## Setup
vf = ast.VectorFunctions
oc = ast.OptimalControl

Args = vf.Arguments
Tmodes = oc.TranscriptionModes
Cmodes = oc.ControlModes
PhaseRegs = oc.PhaseRegionFlags


################################################################################
## System Dynamics
def ODE(mu, ltacc=False):
    irows = 7
    if ltacc != False:
        irows += 3

    args = vf.Arguments(irows)
    r = args.head_3()
    v = args.segment_3(3)
    g = r.normalized_power3() * (-mu)
    if ltacc != False:
        thrust = args.tail_3() * ltacc
        acc = g + thrust
    else:
        acc = g
    return vf.Stack([v, acc])


################################################################################
## Initial Guess Generators
def MakeCircIG(r, thetadeg):
    v = np.sqrt(1.0 / r)
    theta = np.deg2rad(thetadeg)
    IGC = np.zeros((7))
    IGC[0] = np.cos(theta) * r
    IGC[1] = np.sin(theta) * r
    IGC[3] = -np.sin(theta) * v
    IGC[4] = np.cos(theta) * v
    return IGC


def MakeCircTraj(r, thetadeg, tf, n):
    ode = oc.ode_6.ode(ODE(1))
    integ = oc.ode_6.integrator(ode, 0.01)
    IGC = MakeCircIG(r, thetadeg)
    Temp = integ.integrate_dense(IGC, tf, n)
    Traj = []
    for T in Temp:
        TT = np.zeros((10))
        TT[0:7] = T
        TT[7:10] = np.ones((3)) * 0.01
        Traj.append(TT)
    return Traj


################################################################################
## Solver Function
def MultSpaceCraft(Trajs, IStates, SetPointIG, LTacc=0.01, NSegs=75):

    ##Section 1: Create Optimal Control Problem
    ocp = oc.OptimalControlProblem()

    ## create ODE governing all spacecraft
    ode = oc.ode_6_3.ode(ODE(1, LTacc), 6, 3)

    for i, T in enumerate(Trajs):

        ## Create a phase for Each Spacecraft
        phase = ode.phase(Tmodes.LGL5)
        ## Set Initial Guess
        phase.setTraj(T, NSegs)

        ##Use block constant control
        phase.setControlMode(Cmodes.BlockConstant)

        ##Specify that initial state and time are locked at
        ##whatever value is passed to optimizer
        phase.addValueLock(PhaseRegs.Front, range(0, 7))

        ## Bound Norm of Control Vector over the whole phase
        phase.addLUNormBound(PhaseRegs.Path, [7, 8, 9], 0.01, 1.0, 1)

        # Add TOF objective
        phase.addDeltaTimeObjective(1.0)

        ## add phase to the OCP
        ocp.addPhase(phase)

    ####################################################
    # Section 2:
    """
    Adding a Link constraint to enforce that the terminal state and time
    of each phase must be equal to a free state added as LinkParameters of the ocp

    ie: for each phase(i) Xt_i(tf) - Xt_link = 0
    """

    # First we add an initial guess for the linkParams, which we be a free
    # terminal position,velocity and time that all phases must hit
    # The ocp now has 7 link params indexed 0->6
    ocp.setLinkParams(SetPointIG[0:7])

    # Now we need to define the function and varibales needed to express
    # the constraint

    ## The constraint function enforces the equality of two length 7 vectors
    LinkFun = Args(14).head(7) - Args(14).tail(7)

    ## Specifying for each call to collect the x variables indexed
    ## by xlinkvars (position velocity time) at PhaseReg.Back (last state),
    ## these will be the first 7 arguments to each call of LinkFun
    linkregs = [PhaseRegs.Back]
    phasestolink = [[i] for i in range(0, len(Trajs))]
    xlinkvars = [range(0, 7)]

    ## Specifies that for each call, collect the the ocp link vars representing
    ## the free state and forward them to LinkFun, these will be the final 7
    ## arguments for each call
    linkparmavars = [range(0, 7) for i in range(0, len(Trajs))]

    ## combine function and indexing info into LinkConstraint Object and
    ## add it to the phase
    ocp.addLinkEqualCon(LinkFun, linkregs, phasestolink, xlinkvars, linkparmavars)

    ocp.addLinkParamEqualCon(Args(6).head3().dot(Args(6).tail3()), range(0, 6))

    ocp.optimizer.OptLSMode = ast.LineSearchModes.L1
    ocp.optimizer.deltaH = 5.0e-8
    ocp.optimizer.KKTtol = 1.0e-9
    ocp.optimizer.BoundFraction = 0.997
    ocp.optimizer.PrintLevel = 1
    ocp.optimizer.MaxLSIters = 1

    Data = []

    ##################################################################
    # Section 3:
    """
    Now we are going to run an optimization continuation scheme to compute
    the constellation trajectory for each list of initial states of the spacecraft

    """

    for j, Ist in enumerate(IStates):

        ## For each set Initial condtions subsitute the fixed intial conditions
        ## to each phase, Because we locked them, they will be fixed at these values
        ## this avoids having to retranscribe to the problem for every optimize
        for i, phase in enumerate(ocp.Phases):
            phase.subVariables(PhaseRegs.Front, range(0, 7), Ist[i][0:7])

        # force a retranscription peridically to keep problem well conditioned
        # This is not strictly necessary
        if (j > 0) and (j % 8 == 0):
            ocp.transcribe(False, False)

        # Solve before optimizing for the intial run
        if j == 0:
            ocp.solve()
        t0 = time.perf_counter()
        Flag = ocp.optimize()
        tf = time.perf_counter()
        print((tf - t0) * 1000.0)
        if Flag == ast.ConvergenceFlags.NOTCONVERGED:
            ocp.solve_optimize()

        Data.append(
            [[phase.returnTraj() for phase in ocp.Phases], ocp.returnLinkParams()]
        )
    return Data


################################################################################
## Plotting Utilities
def colorScale(x, left=[48, 59, 194], right=[208, 35, 70]):
    return [int(round((x * right[i]) + ((1 - x) * left[i]))) for i in range(3)]


def plotPhaseAndThrottle(tList, name):
    # Take N planar trajectories and calculate angles between them
    angs = [[] for _ in tList]
    for i in range(len(tList[0])):
        base = tList[0][i][0:3] / np.linalg.norm(tList[0][i][0:3])
        for j in range(len(tList)):
            if j == 0:
                angs[j].append(0)
            else:
                unitJ = tList[j][i][0:3] / np.linalg.norm(tList[j][i][0:3])
                angs[j].append(np.arccos(np.dot(base, unitJ)))

    fig = make_subplots(rows=2, cols=1)
    for i, t in enumerate(tList):
        clr = colorScale(i / len(tList))
        fig.add_trace(
            go.Scatter(
                x=[X[6] for X in t],
                y=[A for A in angs[i]],
                mode="lines",
                name="Craft {} Phase".format(i),
                line=dict(color="rgb({},{},{})".format(clr[0], clr[1], clr[2])),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[X[6] for X in t],
                y=[X[7] ** 2 + X[8] ** 2 + X[9] ** 2 for X in t],
                mode="lines",
                name="Craft {} Control".format(i),
                line=dict(color="rgb({},{},{})".format(clr[0], clr[1], clr[2])),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Phase Angle", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Control Magnitude", row=2, col=1)

    fig.show()
    fig.write_html("./{}.html".format(name))


################################################################################
## Main
def main():
    n = 10

    Thetas = np.linspace(20, 180, 20)
    TrajsIG = [
        MakeCircTraj(1, theta, 2.0 * np.pi, 300)
        for theta in np.linspace(0, Thetas[0], n)
    ]
    SetPointIG = TrajsIG[int((n - 1) / 2)][-1][0:7]
    AllIGs = []
    for i, Theta in enumerate(Thetas):
        IStates = [MakeCircIG(1, theta) for theta in np.linspace(0, Theta, n)]
        AllIGs.append(IStates)

    accs = np.linspace(0.015, 0.005, 2)

    timeTrace = []
    for i, a in enumerate(accs):
        Times = []
        Data = MultSpaceCraft(TrajsIG, AllIGs, SetPointIG, a)
        for D in Data:
            SetPoint = D[1]
            Times.append(SetPoint[6] / (2.0 * np.pi))

        timeTrace.append(go.Scatter(x=Times, y=Thetas, mode="lines"))

    fig1 = go.Figure(data=timeTrace)
    fig1.update_xaxes(title_text="Time (ND)")
    fig1.update_yaxes(title_text="Theta (deg)")
    # fig1.show()

    plotTrajs = Data[-1][0]
    plotPhaseAndThrottle(plotTrajs, "Rendezvous")


################################################################################
## Run
if __name__ == "__main__":
    main()
