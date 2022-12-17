Example 4: Space Shuttle Reentry
================================

In this example we will optimize a simple low-thrust spacecraft trajectory with 3 different types of objectives. Then we will
utilize ASSET's costate estimation features to solve the same problem using a semi-direct formulation.

The dynamics for the low-thrust spacecraft are given by eq.1, which assumes a single graviational body centered at the origin of the frame. The control
is then just a pure accelleration that can be applied in any direction. We assume that engine providing this accelleration has a low enough thrust that it does not appreciably
change the spacecraft's mass during the transfer (a common assumption in preliminary trajectory design). As in most of our examples, we will assume that mu =1

.. math::

    \dot{\vec{r}}= \vec{v}
    
    \dot{\vec{v}}= -\mu \frac{\vec{r}}{|r|^2} a^* \vec{u}


We can then implement this model in Asset as an ode_x_u object with just a few lines. It has 6 controls (position,velocity) and 3 control variables (throttle vector).
We then instanatiate the model with mu value of one and eaccellaration value of .02.

.. code-block:: python

    class LTModel(oc.ode_x_u.ode):
        def __init__(self, mu, ltacc):
            Xvars = 6
            Uvars = 3
            ######################################
            args = oc.ODEArguments(Xvars, Uvars)
            r = args.head3()
            v = args.segment3(3)
            u = args.tail3()
            g = r.normalized_power3() * (-mu)
            thrust = u * ltacc
            acc = g + thrust
            ode = vf.stack([v, acc])
            #######################################
            super().__init__(ode, Xvars, Uvars)

    mu = 1
    acc = .02
    ode = LTModel(mu, acc)


With these dynamics we want to compute a transfer from an circular orbit starting on the x-axis at r=1 to a terminal state in a circular orbit of r= 2 on the positive x-axis.
    
.. code-block:: python

    r0 = 1.0
    v0 = np.sqrt(mu / r0)
    rf = 2.0
    vF = np.sqrt(mu / rf)

    X0 = np.zeros((7))
    X0[0] = r0
    X0[4] = v0

    Xf = np.zeros((6))
    Xf[0] = rf
    Xf[4] = vF

A good initial guess for this type of transfer is integrate the initial state with a prgrade thrusting control law until it comes near our desired final state.
This is straight forward to with ASSET's integrator function objects as shown below. 

.. code-block:: python

    integ = ode.integrator(.01, Args(3).normalized() * .8, [3, 4, 5])

    XIG = np.zeros((10))
    XIG[0:7] = X0

    TrajIG = integ.integrate_dense(XIG, 6.4 * np.pi, 100)

    
First we instaate an ode_x_u.integrator object from our ode. The first argument is the default stepsize of the integrator. The second argument is a function 
which will be used to compute the control at each state during the integration, and the third argument is a list of the indices of the state varaibles that 
will be fed to that function. The controller we have specicified here applies the 80 perdecent acceleration along the velocity vector (indices 3,4,5). We then costruct an
appropratiely sized input for our integrator/ode then use the integrate_dense method to generate trajectory with 100 states. The final time was chosen by trial and error but one
could use some of the more advanced features to automate this (see integrator tutoriual).


With our initial guess in hand we can now define our optimization problem with ode_x_u.phase object. We apply our known inital and terminal states as simple boudary
value constraints at the front and back of the trajectory respectively. We also bound the norm of the throttle vector to be between 0.001 and 1 throughout the trajectory.
Note that the lower bound is not 0 as one might expect,because this would make the derivative of the cosntraint infinite when u =0. This is such common problem with this type of control 
parameterization, that the small lower bound has been given name, the leak mass, since it stipluates that the engine is always at least partially on and thus "leaking" mass.

.. code-block:: python

    phase = ode.phase(Tmodes.LGL3, TrajIG, 256)
    phase.addBoundaryValue(PhaseRegs.Front, range(0, 7), X0)
    phase.addLUNormBound(PhaseRegs.Path, [7, 8, 9], .001, 1.0, 1.0)
    phase.addBoundaryValue(PhaseRegs.Back, range(0, 6), Xf[0:6])
    phase.optimizer.PrintLevel = 1

With this constraint settup, we will optimiza the transfer with three different objectives in succession and compare their behavior. For the first run, we will
compute the minimum time transfer. Next we will compute the so called minimum power transfer which minimizes the intrgral of the throttle sqaured. L


