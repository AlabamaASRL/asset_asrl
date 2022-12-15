import numpy as np
import asset_asrl as ast
import matplotlib.pyplot as plt
import time


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
class TwoBody(oc.ode_x_u.ode):
    def __init__(self, P1mu, ltacc=False):
        Xvars = 6
        Uvars = 0
        if ltacc != False:
            Uvars = 3
    
        args = oc.ODEArguments(Xvars, Uvars)
        r = args.head3()
        v = args.segment_3(3)
        g = r.normalized_power3() * (-P1mu)
        if ltacc != False:
            thrust = args.tail_3() * ltacc
            acc = g + thrust
        else:
            acc = g
        ode = vf.stack([v, acc])
        super().__init__(ode, Xvars, Uvars)


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
    ode = TwoBody(1)
    integ = ode.integrator(.01)
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
    ode = TwoBody(1, LTacc)

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
    #Section 2:
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
    ## by xlinkvars (position velocity time) at PhaseRegs.Back (last state),
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

    ocp.optimizer.QPThreads = 8  # Equal to number of physical cores
    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_deltaH(5.0e-8)
    ocp.optimizer.set_KKTtol(1.0e-9)
    ocp.optimizer.set_BoundFraction(0.997)
    ocp.optimizer.PrintLevel = 1
    ocp.optimizer.set_MaxLSIters(1)

    Data = []

    ##################################################################
    #Section 3:
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
        if Flag == ast.Solvers.ConvergenceFlags.NOTCONVERGED:
            ocp.solve_optimize()

        Data.append(
            [[phase.returnTraj() for phase in ocp.Phases], ocp.returnLinkParams()]
        )
    return Data


################################################################################
## Plotting Utilities
def colorScale(x, left=[48, 59, 194], right=[208, 35, 70]):
    return [int(round((x * right[i]) + ((1 - x) * left[i])))/(256) for i in range(3)]


def plotPhaseAndThrottle(tList):
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
    fig, axes = plt.subplots(2, 1, figsize = (12, 8))
    for i, t in enumerate(tList):
        clr = colorScale(i / len(tList))
        x1=[X[6] for X in t]
        y1=[A for A in angs[i]]
        axes[0].plot(x1, y1, color = [(clr[0]), (clr[1]), (clr[2])],
                     label = "S/C "+str(i))
        
        x2=[X[6] for X in t]
        y2=[X[7] ** 2 + X[8] ** 2 + X[9] ** 2 for X in t]
        axes[1].plot(x2, y2, color = [(clr[0]), (clr[1]), (clr[2])])
    axes[0].grid(True)
    axes[0].set_ylabel("Phase Angle (rad)")
    
    axes[1].grid(True)
    axes[1].set_xlabel("Time (ND)")
    axes[1].set_ylabel("Control Magnitude")
    plt.tight_layout()
    axes[0].legend()
    plt.savefig("Plots/MultiSpacecraftOptimization/multispacecraftoptimization.svg",
                dpi = 500)
    plt.show()

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

    for i, a in enumerate(accs):
        Times = []
        Data = MultSpaceCraft(TrajsIG, AllIGs, SetPointIG, a)
        for D in Data:
            SetPoint = D[1]
            Times.append(SetPoint[6] / (2.0 * np.pi))

    plotTrajs = Data[-1][0]
    plotPhaseAndThrottle(plotTrajs)


################################################################################
## Run
if __name__ == "__main__":
    main()

