Multi-Spacecraft Optimization
========================================


Performing a multi-spacecraft optimization with ASSET is virtually identically to the single spacecraft case, except for one new constraint that we will focus on later.
For now, the standard problem procedure is the same, where we need to import everything from ASSET with our standard conventions, as well as some plotting tools for the output.

.. code-block:: python

    import numpy as np
    import asset_asrl as ast
    import matplotlib.pyplot as plt
    import time


    ################################################################################
    ## Setup
    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments
    

Now that we have all of the tools imported from ASSET, as always we need to identify our dynamics for the problem. Here, for this simple case,
we have assumed two-body gravity dynamics, plus each spacecraft will be equipped with a low thrust propulsion system. Writing the dynamics is the same as in previous
examples, where we will use the ASSET VectorFunctions to construct our system of ODEs.

.. code-block:: python

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
            v = args.segment3(3)
            g = r.normalized_power3() * (-P1mu)
            if ltacc != False:
                thrust = args.tail3() * ltacc
                acc = g + thrust
            else:
                acc = g
            ode = vf.stack([v, acc])
            super().__init__(ode, Xvars, Uvars)

Here the dynamics are simply a function of the 6 state variables (position and velocity). The :code:`ltacc` flag
determines if we are incorporating the low thrust acceleration into the function or not. We make this choice so that if we wish to propagate the spacecraft
without low thrust and purely gravity as the only acceleration. If :code:`ltacc` is given a value, we will include a low thrust acceleration into the model, as well as the
normal vector of the low thrust, where we add 3 to the input rows of the state. This will let us control the direction of the thrust for the optimization call.

As is standard procedure for optimization calls, we have to generate initial guesses for our problem. A very easy case to use is that of an initially circular orbit for our spacecraft.
Assuming that these are planar circular orbits, we can generate them with:

.. code-block:: python

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

:code:`MakeCircIG` is responsible for returning the position and velocity of the spacecraft for a specified radius of :code:`r`, as well as a given
true anomaly :code:`thetadeg`. This is also the case for :code:`MakeCircTraj`, which will call :code:`MakeCircIG` when it is determining the initial states
of the spacecraft. To simplify our design flow, :code:`MakeCircTraj` initializes the ode for each spacecraft, through the ASSET optimal control interface
and integrates the trajectory out for the given time :code:`tf` (:code:`n` determines the number of points to use for the output trajectory).
:code:`MakeCircTraj` returns the integrated trajectory for the time :code:`tf`, and will have a number of states equal to :code:`n`.

Now, we have our dynamics, as well as a method to produce initial guesses for the multi-spacecraft optimization problem. The next step to do
is to define a function to wrap our optimization calls in. This is similar to what we have done in the previous example, :ref:`Zermelo's Problem`, except now we will have
an extra LinkConstraint that will enforce that each final states of the spacecraft must be equal to a desired free state that we will add.
For now we will show the function that handles all this in three sections, with the final full function definition at the end of the example.

.. code-block:: python

    ################################################################################
    ## Solver Function
    def MultSpaceCraft(Trajs, IStates, SetPointIG, LTacc=0.01, NSegs=75):

        ##Section 1: Create Optimal Control Problem
        ocp = oc.OptimalControlProblem()

        ## create ODE governing all spacecraft
        ode = TwoBody(1, LTacc)

        for i, T in enumerate(Trajs):

            ## Create a phase for Each Spacecraft
            phase = ode.phase("LGL5")
            ## Set Initial Guess
            phase.setTraj(T, NSegs)

            ##Use block constant control
            phase.setControlMode("BlockConstant")

            ##Specify that initial state and time are locked at
            ##whatever value is passed to optimizer
            phase.addValueLock("Front", range(0, 7))

            ## Bound Norm of Control Vector over the whole phase
            phase.addLUNormBound("Path", [7, 8, 9], 0.01, 1.0, 1)

            # Add TOF objective
            phase.addDeltaTimeObjective(1.0)

            ## add phase to the OCP
            ocp.addPhase(phase)

The first section of :code:`MultiSpaceCraft` is very similar to the previous definitions for ASSET optimization routines.
It takes as arguments the list of initial circular orbits, :code:`Trajs`.
The next input :code:`IStates` is the list of initial states for each of the spacecraft. Each spacecraft will also need to be given a specific state
to target for the final end stand, :code:`SetPointIG`. Lastly, the low thrust acceleration is assigned a non-dimensional value :code:`LTacc` of .01 and the number of segments for each trajectory :code:`Nsegs`
is given a value of 75. the rest of this code is the same as we have seen in previous examples to establish the base of the optimization routine.

The next section of code continues the above function. Here we need to define the link constraint that will enforce that each spacecraft
reach some initial final free state.

.. code-block:: python

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

    # Forward the back state in each phase and the linkParams to the function
    for i in range(0,len(Trajs)):
        ocp.addLinkEqualCon(LinkFun,[(i,"Back",range(0,7),[],[])],range(0,7))

    ocp.addLinkParamEqualCon(Args(6).head3().dot(Args(6).tail3()), range(0, 6))

    ocp.optimizer.set_OptLSMode("L1")
    ocp.optimizer.set_deltaH(5.0e-8)
    ocp.optimizer.set_KKTtol(1.0e-9)
    ocp.optimizer.set_BoundFraction(0.997)
    ocp.optimizer.PrintLevel = 1
    ocp.optimizer.set_MaxLSIters(1)

    Data = []

First we must choose which part of each state we desire to enforce this constraint for. Clearly, we wish each spacecraft to arrive at some
final position, velocity, and time, so we set the link parameters to be the point we passed in :code:`SetPointIG`. It is length 7, and following our convention
the first 3 are position, the next 3 are velocity, and 7th variable is the desired final time. We assign the link parameter to the optimal control interface with :code:`ocp.setLinkParams(SetPointIG[0:7])`.
Now we will construct a VectorFunction representing the constraint. To construct this function we define a variable :code:`LinkFun`, wich is simply subtracting the last 7 variables
from our :code:`Args` (our desired final point), and the first 7 (our initial spacecraft state).

With this done, we need a way to collect all the variables as each step to tie the phases together. We know we want the last states linked together, so we assign :code:`linkregs` to be the PhaseRegionFlag :code:`PhaseRegs.Back`.
Now we set all the phases that need to be linked together (all of them), with :code:`phasestolink` and tell :code:`xlinkvars` that we want the first 7 variables of each. The last step before we add the constraint to the problem,
is to create an argument that specifies we want the first 7 variables of **each** trajectory from :code:`Trajs`.

All of this comes together in :code:`ocp.addLinkEqualCon(LinkFun, linkregs, phasestolink, xlinkvars, linkparmavars)`, creating the link constraint for the optimization problem.
The last bit of this section is setting the linesearch mode (:code:`ocp.optimizer.OptLSMode`), as well as tolerances on the optimization problem.

The very last section of the code neccessary for the multi-spacecraft optimization problem is to actually run the optimizer! We will need to do this for every initial state we pass into the problem, with each state representing a spacecraft in the constellation.


.. code-block:: python

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
            phase.subVariables("Front", range(0, 7), Ist[i][0:7])

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

The first :code:`for` loop in this section assigns the values of our desired initial conditions into the :code:`ocp.Phases` interface.
The actual optimization code that executes the solution is likely the simplest bit of code in this problem (as we know constructing a problem statement in a logical manner can be the hardest part of optimization).
We run a :code:`ocp.solve` on each initial state to make our initial guess better by satisfying the constraints before we even begin optimizing. We are also curious about the total time to solve each problem, so we set
a few timers with the :code:`Python::time` library. We run the optimize call between the timers so we know how much time is taken up by the optimizer. Lastly, we check if at the end of the optimization
if the :code:`ast.ConvergenceFlags` is satisfied, and if not we run :code:`ocp.solve_optimize()` to solve and optimize the problem again. Then we save the data in a format that will make it easier to plot.

Below is the code we use to plot, but the user can use whatever they are most comfortable with for their own purposes.

.. code-block:: python

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
        plt.savefig("Plots/MultiSpacecraftOptimization/multispacecraftoptimization.pdf",
                    dpi = 500)
        plt.show()


Bringing everything together into the main function of the problem, we create out initial guesses, determine our final point, and call the :code:`MultiSpaceCraft` function.
We decide that we want 10 spacecraft and we will space them all out along the same orbit in 20 degree increments, up to 180 degrees. These will be our initial states for the optimization problem.

.. code-block:: python

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


Our initial guess for the final point to target is taken to be the middle spacecraft's last state at the end of its initial trajectory in :code:`SetPointIG`. All of our initial states are generated in the next :code:`for`
loop, where we make sure that every initial state is corresponding to a circular orbit. We are interested in how the low thrust acceleration of the vehicle affects the ability for our spacecraft to rendezvous to the desired final state,
so we create a list of various non-dimensional accelerations in :code:`accs`. Now all we do is iterate over the list of accelerations and call our :code:`MultiSpaceCraft` function with all of the required inputs.
What we get is an optimization problem that simultaneously solves for the optimal control of all spacecraft to converge on the final point.

.. figure:: _static/multispacecraftoptimization.svg
    :width: 100%
    :align: center

The top plot shows the spacecraft converging to the final point, indicated by the phase angles between the spacecraft decreasing towards 0. The bottom plot shows the complex control histories of 10 spacecraft
manuevering in tandem to satisfy a given objective. Any further analysis is outside of the scope of this tutorial and is left to the reader.

Full Code:
##########

.. code-block:: python

    import numpy as np
    import asset_asrl as ast
    import matplotlib.pyplot as plt
    import time


    ################################################################################
    ## Setup
    vf = ast.VectorFunctions
    oc = ast.OptimalControl

    Args = vf.Arguments


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
            v = args.segment3(3)
            g = r.normalized_power3() * (-P1mu)
            if ltacc != False:
                thrust = args.tail3() * ltacc
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
            phase = ode.phase("LGL5")
            ## Set Initial Guess
            phase.setTraj(T, NSegs)

            ##Use block constant control
            phase.setControlMode("BlockConstant")

            ##Specify that initial state and time are locked at
            ##whatever value is passed to optimizer
            phase.addValueLock("Front", range(0, 7))

            ## Bound Norm of Control Vector over the whole phase
            phase.addLUNormBound("Path", [7, 8, 9], 0.01, 1.0, 1)

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

        # Forward the back state in each phase and the linkParams to the function
        for i in range(0,len(Trajs)):
            ocp.addLinkEqualCon(LinkFun,[(i,"Back",range(0,7),[],[])],range(0,7))

        ocp.addLinkParamEqualCon(Args(6).head3().dot(Args(6).tail3()), range(0, 6))

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
                phase.subVariables("Front", range(0, 7), Ist[i][0:7])

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
