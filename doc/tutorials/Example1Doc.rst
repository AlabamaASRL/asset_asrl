Optimal Control Example 1 (Quadratic Control)
==========================

As a first introduction to the optimal control module, we will solve a simple quadratic
state control problem with a known analytic solution $\vec{S}$.


.. math::
    
    \vec{S} = [\vec{X},t,\vec{U}]^T = [x,t,u]^T
    \\
    \\
    J(X) = \int_0^{t_f} (u^2 + x*u + \frac{5}{4} x^2) dt

    \dot{x} = \frac{x}{2} + u
    
    x(0)= 1
    
    t_f  = 1
    


.. code-block:: python

    import numpy as np
    import asset as ast
    import matplotlib.pyplot as plt
    
    vf = ast.VectorFunctions
    oc = ast.OptimalControl
    Args = vf.Arguments
    Tmodes = oc.TranscriptionModes
    PhaseRegs = oc.PhaseRegionFlags
    
    class Example1ODE(oc.ode_x_u.ode):
        def __init__(self):
            Xvars = 1
            Uvars = 1
            Ivars = Xvars + Uvars + 1
            '''
            XtU = [x,t,u]
            xdot = x/2 + u
            '''
            #################
            args = vf.Arguments(Ivars)
            x=args[0]
            u=args[2]
            xdot = .5*x + u
            super().__init__(xdot,Xvars,Uvars)
    
        class obj(ast.ScalarFunctional):
            def __init__(self):
                args = Args(2)
                x=args[0]
                u=args[1]
                obj = u*u + x*u + 1.25*x*x
                super().__init__(obj)
    

To solve the previously defined model, we instantate an instanace which we will call "ode".
We then construct an initial guess "TrajIG", with constant state and control over the interval between
t0 and tf. Next we instatiate a "Phase" object from our ode. The argument to ode.phase is the enumerator indicating
the transcription method that will be used to approximate dynamics along this phase of the trajectory. Valid options are 
constained inside oc.TranscriptionModes (Tmodes). For this example we will use the LGL5 collocation method. Next we set the phases's
Trajectory and specify the number of LGL segments to be used. We then apply the two boundary value constraints specified in the problem definition.
The first argument to addBoundaryValue is an enumerator specifying where along the phase the constraint will be added. Valid options are contained within
the object oc.PhaseRegionFlags (PhaseRegs). The initial conditions are applied at PhaseRegs.Front. The next argument is then a vector of integers specifiying


.. code-block:: python

    ode = Example1ODE()
    
    x0 = 1.0
    t0 = 0.0
    tf = 1.0
    u0 = .01
    
    nsegs = 100
    method = Tmodes.LGL5
    
    TrajIG = [[x0,t,u0] for t in np.linspace(t0,tf,100)]
    
    phase = ode.phase(method)
    phase.setTraj(TrajIG,nsegs)
    phase.setControlMode(oc.HighestOrderSpline)
    phase.addBoundaryValue(PhaseRegs.Front,[0,1],[x0,t0])
    phase.addBoundaryValue(PhaseRegs.Back, [1],  [tf])
    phase.addIntegralObjective(Example1ODE.obj(),[0,2])
    
    phase.optimize()
    
    Traj = phase.returnTraj()
    
    CTraj= phase.returnCostateTraj()
