Orbit Family Continuation
====================================

A use case familiar to all astrodynamicists is the generation of orbit families via continuation from an initial orbit.
ASSET makes reconvergence of slightly perturbed solutions quick and simple by using common python idioms.
The scenario we will investigate is in the Earth-Moon CR3BP, where we want to build the L1 Planar and Northern Halo families from two approximate initial conditions.



Problem Setup
-------------

All python scripts begin by importing necessary external packages.
Obviously, we will need ASSET, but we will also use numpy for arrays and matplotlib for graphing.

.. code-block:: python

    import asset_asrl as ast
    from asset_asrl.Astro.AstroModels import CR3BP 
    import asset_asrl.Astro.Constants as c
    import numpy as np
    import copy
    import matplotlib.pyplot as plt

Next, we will define shorthand abbreviations for some of ASSET's common functionality.
:code:`OptimalControl` and :code:`VectorFunctions` are the two primary submodules, and it makes it easier on us if we don't have to type the full name every time.
:code:`PhaseRegionFlags` and :code:`TranscriptionModes` are enumerators used to specify behavior when we're constructing the solver.
If they are unfamiliar, you should go back to the introductory tutorials.

.. code-block:: python

    ################################################################################
    # Setup
    oc = ast.OptimalControl
    vf = ast.VectorFunctions

    phaseRegs = oc.PhaseRegionFlags
    tModes = oc.TranscriptionModes


Now we define some system parameters.
These quantities are known ahead of time and will not change during the course of continuation or optimization.
For our test problem, we are analyzing the Earth-Moon system, so the appropriate masses and characteristic distance go here.
Also, we define a delta-t for our integrator step.

.. code-block:: python

    ################################################################################
    # Constants
    muE = c.MuEarth  # Earth gravitational parameter
    muM = c.MuMoon # Moon gravitational parameter
    lstar = c.LD #Earth-Moon characteristic distance (m)

    dt = 3.1415 / 10000


Given the known system parameters, we can construct the dynamical model.
ASSET is written generically from a low level, but we tailor it to astrodynamics by providing some pre-built models, such as the CR3BP.
The instance of the governing ordinary differential equations (ODE) for the CR3BP model is instantiated with the gravitational parameters of the
Earth and Moon (in :math:`\frac{m^3}{s^2}`), along with a characteristic distance (in this case its the average distance between the Earth and Moon in meters).
The non-dimensional gravity parameter, :math:`\mu`, of the system is stored within the CR3BP model as a member.
Furthermore, the ODE has an integrator associated with it which we can instantiate with a fixed time step dt.
The :code:`dt` here is the maximum allowable time step whenever the integrator is invoked.

.. code-block:: python

    ################################################################################
    # System Dynamics
    
    ode = CR3BP(muE, muM, lstar)

    mu = ode.mu

    # Create integrator (= propagator)
    odeItg = ode.integrator(dt)



Constructing a Generic Periodic Solver
--------------------------------------

Astrodynamicists are certainly aware that the characteristic that defines periodicity in the CR3BP is an orthogonal crossing of the x-z plane.
In order to have good code re-usability, let's define a function that will try to solve a periodic orbit given some inital guess of state and duration.
Here's the function signature we will use: :code:`def solvePeriodic(ig, tf, fixInit=[0, 1, 2]):`
The first parameter, :code:`ig`, is a 7-element vector with :math:`x`, :math:`y`, :math:`z` position in the first elements, :math:`v_x`, :math:`v_y`, :math:`v_z` in the next 3 elements, and initial time :math:`t_0` in the last position.
The second parameter, :code:`tf`, will be the best guess for the orbit's half-period.
The third parameter, :code:`fixInit`, defines which terms of the initial guess are not to be changed during the solve.
For example, if you want a planar orbit, you would set :code:`ig[2] = 0` and :code:`fixInit = [2]` to fix the second element of the guess.
Or, suppose you just want to find some orbit with a certain :math:`v_y`.
You could pass :code:`fixInit = [4]` to lock the initial velocity but allow the starting position to change.

Given these arguments, the outline for solution is as follows:

1. Integrate the initial guess state to get an initial trajectory guess.
2. Create an optimal control phase and initailize it with the trajectory guess.
3. Add constraints to the phase that are appropriate for periodicity.
4. Solve for the orbit.
5. Return the orbit.

Here is the python implementation, with discussion below:

.. code-block:: python

    ################################################################################
    # Solve for periodic orbit using initial guess ig
    def solvePeriodic(ig, tf, ode, odeItg, fixInit=[0, 1, 2]):
        # 1: Integrate initial guess
        steps = 1000
        trajGuess = odeItg.integrate_dense(ig, tf, steps)

        # 2: Create optimal control phase and assign guess
        odePhase = ast.CR3BP.phase(ode, tModes.LGL3)  # LGL-3 collocation
        odePhase.Threads = 8  # Equal to number of physical cores

        nSeg = 150  # number of segments
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


First, the guess is integrated to the specified final time in a given number of steps.
The :code:`steps` argument given here can override the previous :code:`dt` only if :math:`\frac{t_f - t_0}{steps} < \delta t`.
Thus, :code:`trajGuess` is a list of states that compose a trajectory.

The optimal control phase is associated with the CR3BP ODE, and is initialized to use an LGL3 collocation scheme via the enumerator :code:`tModes`.
When we pass in :code:`trajGuess`, we need to tell the phase how many LGL3 arcs to split it into.
This parameter must be high enough to generate an accurate approximation of the real dynamics, but it can negatively impact runtime if it is too large.
In general, basic trial and error is sufficient to tune this parameter.
Lastly, the phase is told to use 8 CPU threads.
You could scale this term up or down to fit your system.

Now we set the constraints that enforce periodicity.
Terms passed in :code:`fixInit` are handled first, by adding a boundary value at the front of the trajectory.
Then come the required constraints.
At the beginning of any periodic orbit, the y position and the velocity in the x and z directions must be zero to have an orthogonal crossing of the x-z plane.
We define the initial time to be zero as well.
At the end of the trajectory, the same conditions must hold, except that the final time cannot be zero for a non-trivial solution.

Before solving the trajectory, we must establish what constitutes a valid solution by setting a convergence tolerance.
In this scenario, the optimization problem only consists of equality constraints, so we set the value of :code:`EContol`.
Then a call to :code:`odePhase.solve()` runs everything we set up.

All that remains is to extract the result, which is done via :code:`returnTraj()`.



Performing Continuation
-----------------------

In order to generate an orbit family, it is not sufficient to solve only *one* orbit, so let's create another function that performs continuation by calling :code:`solvePeriodic` over a range of inputs.
Rudimentary continuation increments some parameter of the orbit, and then re-solves for a new orbit with the different parameter.
Also, most orbit families don't go on forever, so we need a way to stop the progression.
With these factors in mind, we'll start by defining the function signature.
Since :code:`continue` is a reserved keyword in python, we'll abbreviate our function name to :code:`contin`.
Thus, our function is :code:`contin(ig, tf, cIdx, dx, lim, fixInit)`.
The first two arguments, :code:`ig` and :code:`tf` are our initial state and time guesses, just like above.
The next three arguments define the stepping and termination of the continuation scheme.
:code:`cIdx` is the index of the variable we are changing at each step.
To increment :math:`x`, you would pass :code:`cIdx = 0`.
:code:`dx` is how much to increment the :code:`cIdx` 'th term on each iteration, and :code:`lim` is the value at which to terminate the continuation.
Lastly, the :code:`fixInit` argument comes at the end since it has a default value, and it can be used if there are elements that you definitely don't want to change during the continuation.

With the arguments established, the code is presented with discussion below:

.. code-block:: python

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

As you can see, continuation can be as simple as wrapping a :code:`solvePeriodic` call in a while loop that terminates at the given :code:`lim`.
A guess for the current step is obtained from the previous step by pulling it off the end of the :code:`trajList` with python's negative indices.
Then, the specified term is incremented and the :code:`solvePeriodic` function handles the rest.
Of course, this approach doesn't have any error handling if an orbit doesn't converge, but we've shown that a first-pass approximation can be implemented with very basic knowledge of python capabilities.



Running and Plotting
--------------------

Since we've put in the work up front to produce functions that capture the generic concepts of periodicity and continuation, calculating some specific orbit family can be done with minimal code.
First, here's a quick plotting function to graph the list of trajectories we expect to recieve from :code:`contin`.

.. code-block:: python

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

We'll skip discussing this function in detail since matplotlib has it's own documentation.

Now, on to what we promised from the start, L1 Lyapunovs:

.. code-block:: python

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

    plotTrajList(tlp)

We pull an initial guess from any reputable source (e.g. Grebow_), and hot-start the continuation with a preliminary solve.
In this case, we are reducing the inital :math:`x` with each step, as indicated by :code:`cIdx=0` and :code:`dx=-0.001`.
All three initial positions are implicitly fixed by the default value of :code:`fixInit`; this choice will keep solutions in-plane and will ensure we don't solve for the same trajectory twice.
Do note that the continuation limit is set such that we do not obtain the *full* family of Lyapunovs.
A smarter continuation scheme would be necessary to converge the extreme orbits.
Also, we do a bit of trickery with the plotting.
Since it's more stable to solve for half-orbits, we duplicate the trajectory over the x-z plane so that we see the full orbit.

.. figure:: _static/Lyapunov.svg
    :width: 100%
    :align: center

.. _Grebow: https://engineering.purdue.edu/people/kathleen.howell.1/Publications/Masters/2006_Grebow.pdf

The code for L1 Northern Halos is almost identical, save for the initial conditions.
One notable change is the explicit definition of :code:`fixInit`.
In this case, we allow :math:`x` to be adjusted by the solver as we increment :math:`z` so that we follow the correct shape of the family.
Again, we truncate early.

.. code-block:: python

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

    plotTrajList(tlp)

.. figure:: _static/Halo.svg
    :width: 100%
    :align: center

Full Code
#########

.. code-block:: python

    import asset_asrl as ast
    from asset_asrl.Astro.AstroModels import CR3BP
    import asset_asrl.Astro.Constants as c
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
    muE = c.MuEarth  # Earth gravitational parameter
    muM = c.MuMoon # Moon gravitational parameter
    lstar = c.LD #Earth-Moon characteristic distance (m)

    dt = 3.1415 / 10000

    ################################################################################
    # System Dynamics
    
    ode = CR3BP(muE, muM, lstar)

    mu = ode.mu

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
            "Front", [3, 6], [0.0, 0.0]  # Initial vx, t = 0
        )
        odePhase.addBoundaryValue(
            "Back", [1, 3, 5], [0.0, 0.0, 0.0]  # Final y, vx, vz = 0
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
    def plotTrajList(tList, name, proj = False):
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