Integrator Tutorial
===================

In this section, we will describe usage of asset's integrator object. To 
start we will define an ODE which we want to integrate using what we have learned from the
previous two sections. As concrete example we use a ballistic two body
model whose state variables are the position and velocity relative to a central
with gravitational parameter is mu. This ODE may be defined as shown below. 
For this example,we will be working in canonical units where mu=1.


.. code-block:: python
	
	import asset_asrl as ast
	import numpy as np
	import matplotlib.pyplot as plt


	vf        = ast.VectorFunctions
	oc        = ast.OptimalControl
	Args      = vf.Arguments


	class TwoBody(oc.ODEBase):
            def __init__(self,mu):
        
                XVars = 6
                XtU = oc.ODEArguments(XVars)
        
                R,V  = XtU.XVec().tolist([(0,3),(3,3)])
        
                G = -mu*R.normalized_power3()
        
                Rdot = V
                Vdot = G

                ode = vf.stack([Rdot,Vdot])
                super().__init__(ode,XVars)
        

    


Initialization
##############

To start, we can create an integrator for our ODE by using the .integrator method.
At initialization we can specify the integration scheme as a string as well as the
default step size to be used. At the moment we provide the support adaptive Dormand Prince
integration schemes of order 8(7) and 5(4). These are specified by the strings "DOPRI87" or
"DOPRI54". If no string is specified, the integration we will default to "DOPRI87". The second
argument, DefStepSize is first trial step size in the adaptive integration algorithm, which will then
adjust the step size up or down to meet specified error tolerances. By default, we set the
integrator'ss minimum and maximum allowable step sizes to be 10000 times smaller and larger than
the default step size. However, you can override this by manually setting all step sizes using the
.setStepSizes method.


.. code-block:: python
    
    TBode = TwoBody(1.0)

    DefStepSize = .1

    #Initialization
    TBInteg = TBode.integrator("DOPRI87",DefStepSize)
    TBInteg = TBode.integrator("DOPRI54",DefStepSize)
    TBInteg = TBode.integrator(DefStepSize)  # Default is DOPRI87


    print("Default StepSize    = DefStepSize = "      , TBInteg.DefStepSize) 
    print("Default MinStepSize = DefStepSize/10000 = ", TBInteg.MinStepSize) 
    print("Default MaxStepSize = DefStepSize*10000 = ", TBInteg.MaxStepSize) 

    MinStepSize = 1e-6
    MaxStepSize = 1.0

    ## Set def,min, and max step sizes manually
    TBInteg.setStepSizes(DefStepSize,MinStepSize,MaxStepSize)

Both integration schemes use the standard absolute and relative tolerance
metrics to asses the accuracy of steps and update the step size adaptively.
By default we set the absolute tolerance on all state variables equal to 1.0e-12
and the relative tolerance to 0.0. This usually works well for well scaled dynamics 
equations such as our two-body model with mu = 1. However, you can set them manually
yourself using the setAbs/RelTol methods as shown below.


.. code-block:: python
    
    TBode = TwoBody(1.0)
    DefStepSize = .1
    TBInteg = TBode.integrator(DefStepSize)  # Default is DOPRI87

    print("Default AbsTols = [1.0-12...] = ",TBInteg.getAbsTols())
    print("Default RelTols = [1.0-12...] = ",TBInteg.getRelTols())

    AbsTol      = 1.0e-13
    RelTol      = 0

    # Set tolerances uniformly for all state variables
    TBInteg.setAbsTol(AbsTol)
    TBInteg.setRelTol(RelTol)

    AbsTols = np.array([1,1,1,3,3,3])*1.0e-13
    RelTols = np.array([0,0,0,1,1,1])*1.0e-9
    # Set tolerances individually for each state variables
    TBInteg.setAbsTols(AbsTols)
    TBInteg.setRelTols(RelTols)


Integration
###########

Now that we have covered initializing integrators, lets examine how we actually
use them. By far the most used methods are integrate and integrate_dense. Both methods,
take as the first input a full-state vector containing the initial state, time, controls, and
parameters as well as the final time that we wish to integrate these initial inputs to.
 
The .integrate() method  integrates this initial full-state input vector to final time tf and returns just the full-state at the final time.
integrate_dense takes the same inputs but returns all intermediate full-states 
calculated by the integrator as single python list. We also call provide integrate_dense 
with an additional arguments specifying that we would like to return N evenly spaced steps
between t0 and tf rather than the exact steps taken by the solver. These n states and controls
will be calculated from the exact steps taken by the integrator using a fifth order interpolation method. 
For the DOPRI54 method, interpolated states have effectively
the exact same error as the true steps taken by the integrator. However, for the DOPRI87 method, interpolated states
will have the larger locals error owing the difference in order between the integration and interpolation. In practice
the maximum local error at any point along the trajectory is typically 2 orders of magnitude larger than the integration tolerances. 


.. code-block:: python

    TBode = TwoBody(1.0)
    DefStepSize = .1
    TBInteg = TBode.integrator(DefStepSize)  


    r  = 1.0
    v  = 1.1
    t0 = 0.0
    tf = 10.0
    N  = 1000

    X0t0 = np.zeros((7))
    X0t0[0]=r
    X0t0[4]=v
    X0t0[6]=t0

    # Just the final full-state
    Xftf = TBInteg.integrate(X0t0,tf)


    TrajExact  = TBInteg.integrate_dense(X0t0,tf)
    TrajInterpN = TBInteg.integrate_dense(X0t0,tf,N)



Event Detection
###############

We can also pass a list of events to be detected during the integration. An single 
event is defined as a tuple consisting of: An asset scalar function whose zeros
determine the locations of the event, a direction indicating whether we want to track ascending,descending or all zeros, and A stop code
signifying whether integration stop after encountering a zero. The scalar function should take the same arguments as the underlying ODE.
The direction flag should be set to 0 to capture all zeros,-1 to capture only zeros where the function value is decreasing, or 1 to capture
zeros where it is increasing. The stopcode should be set to 0 or False if you do not want an event to stop integration. To stop after 1 occurrence,
stopcode can be set to 1 or True. The stopcode can also be set to any positive integer, in which case it specifies that the number of zeros to be encountered
before stopping. When events are appended to an integration call, in addition to the normal return value, a list of lists of the exact full-states where each event occurred is
also returned. As an example, the code below will calculate the apopses and periapses of an orbit, and stop after both have been found. Exact roots
of events are found using a newton raphson method applied to the fifth order spline contnious rperesentation of the trajectory. The root tolerance
and maximum newton iterations may be specified by modifying the EventTol and MaxEvent Iters firlds of the integrator. These default, to 10 and 1e-6 respectively.


.. code-block:: python

    r  = 1.0
    v  = 1.1
    t0 = 0.0
    tf = 100.0
    N  = 1000


    X0t0 = np.zeros((7))
    X0t0[0]=r
    X0t0[4]=v
    X0t0[6]=t0

    def ApseFunc():
        R,V = Args(7).tolist([(0,3),(3,3)])
        return R.dot(V)

    direction = -1
    stopcode = False
    ApoApseEvent  = (ApseFunc(),direction,stopcode)


    direction = 1
    stopcode = False
    PeriApseEvent  = (ApseFunc(),direction,stopcode)


    direction = 0
    stopcode  = 2  # Stop after finding 2 apses
    AllApseEvent  = (ApseFunc(),direction,stopcode)


    Events = [ApoApseEvent,PeriApseEvent,AllApseEvent]


    TBInteg.EventTol =1.0e-10
    TBInteg.MaxEventIters =12

    ## Just append event list to any normal call
    Xftf, EventLocs = TBInteg.integrate(X0t0,tf,Events)

    Traj, EventLocs  = TBInteg.integrate_dense(X0t0,tf,Events)

    Traj, EventLocs  = TBInteg.integrate_dense(X0t0,tf,N,Events)

    #EventLocs[i] will be empty if the event was not detected

    ApoApseEventLocs  = EventLocs[0]
    ApoApse =ApoApseEventLocs[0]

    PeriApseEventLocs = EventLocs[1]
    PeriApse =PeriApseEventLocs[0]

    # Or
    AllApseEventLocs  = EventLocs[2]
    ApoApse  = AllApseEventLocs[0]
    PeriApse = AllApseEventLocs[1]



Derivatives
###########

In asset integrators themselves are vector functions, and have analytic first and second
derivatives. The input arguments for the integrator when used as a vector function consists of
the full-state to be integrated and the final time tf, and the output is the full-state at time tf.
For example, calling compute as shown below is equivalent to the normal integrate call. This also means
that we can calculate the jacobian and adointhessian as well.

.. code-block:: python

    r  = 1.0
    v  = 1.1
    t0 = 0.0
    tf = 20.0


    X0t0 = np.zeros((7))
    X0t0[0]=r
    X0t0[4]=v
    X0t0[6]=t0

    X0t0tf = np.zeros((8))
    X0t0tf[0:7]=X0t0
    X0t0tf[7]=tf



    Xftf = TBInteg.integrate(X0t0,tf)

    # Same as above
    Xftf = TBInteg.compute(X0t0tf)

    Jac =  TBInteg.jacobian(X0t0tf)
    Hess = TBInteg.adjointhessian(X0t0tf,np.ones((7)))


We should note that the jacobian of an integrator is the same as the state transition matrix (STM).
Since calculation of an ODE's state transition matrix (STM), is critical to the assessing
the stability of periodic orbits, we also provide methods to calculate the STM through the integrator
interface using the integrate_stm methods, which can be used as shown below.


.. code-block:: python
    
    Xftf,Jac = TBInteg.integrate_stm(X0t0,tf)

    ## With Events

    Xftf,Jac, EventLocs = TBInteg.integrate_stm(X0t0,tf,Events)

Parrallel Integration
#####################

Finally, for all previously discussed .iintegrate methods, we have corresponding multi-threaded _parallel
version which will integrate lists of initial conditions and final times in parallel. In each case rather
than passing a single initial state and final time we pass a lists of each. The outputs to the call will then be list
of length n containing the outputs of the regular non-parallel method for the ith input state and time.

.. code-block:: python

    n = 100
    nthreads = 8

    X0t0s =[X0t0]*n
    tfs   =[tf]*n


    Xftfs = TBInteg.integrate_parallel(X0t0s,tfs,nthreads)

    Xftf_Jacs = TBInteg.integrate_stm_parallel(X0t0s,tfs,nthreads)

    Xftf_Jac_EventLocs = TBInteg.integrate_stm_parallel(X0t0s,tfs,Events,nthreads)

    Trajs  = TBInteg.integrate_dense_parallel(X0t0s,tfs,nthreads)

    Traj_EventLocs  = TBInteg.integrate_dense_parallel(X0t0s,tfs,Events,nthreads)

    for i in range(0,n):
    
        Xftf = Xftfs[i]
        Xftf,Jac = Xftf_Jacs[i]
        Xftf,Jac,EventLocs = Xftf_Jac_EventLocs[i]
    
        Traj = Trajs[i]
        Traj,EventLocs = Traj_EventLocs[i]


Local Control Laws
##################

In the previous examples we only examined how to integrate ODE's with no control,
or constant controls, but often time we need to compute controls as a function
of the local state or time. We can do this in asset by initializing our integrator with
a control law. As an example,lets reuse our twobodyLT ode from the ODEGuide section.

.. code-block:: python

    class TwoBodyLTODE(oc.ODEBase):
    
        def __init__(self,mu,MaxLTAcc):
        
            XVars = 6
            UVars = 3
        
        
            XtU = oc.ODEArguments(XVars,UVars)
        
            R,V  = XtU.XVec().tolist([(0,3),(3,3)])
            U = XtU.UVec()
        
        
            G = -mu*R.normalized_power3()
            Acc = G + U*MaxLTAcc
        
            Rdot = V
            Vdot = Acc
        
            ode = vf.stack([Rdot,Vdot])
        
            super().__init__(ode,XVars,UVars)
        
We will add a control to the integrator specifying that throttle should be 80%
of the maximum and aligned with the spacecraft's instantaneous velocity vector. We do this by first writing
an ASSET vector function , that is assumed to take only the velocity as arguments and outputs
the desired control vector. We can then pass this to integrator constructor along with a list
specifying the indices full-state variables we want to forward to out control law. In this case it
is [3,4,5] which are the velocity variables as we have defined in our ODE.
This control law will now be applied to all of our integrations.

.. code-block:: python

    def ULaw(throttle):
        V = Args(3)
        return V.normalized()*throttle


    ode = TwoBodyLTODE(1,.01)

    integNoUlaw = ode.integrator("DOPRI87",.1)
    integULaw   = ode.integrator("DOPRI87",.1,ULaw(0.8),[3,4,5])



    r  = 1.0
    v  = 1.1
    t0 = 0.0
    tf = 20.0


    X0t0U0 = np.zeros((10))
    X0t0U0[0]=r
    X0t0U0[4]=v
    X0t0U0[6]=t0        


    TrajNoULaw = integNoUlaw.integrate_dense(X0t0U0,tf)

    TrajULaw   = integULaw.integrate_dense(X0t0U0,tf)

