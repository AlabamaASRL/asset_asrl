#########################
Optimal Control Utilities
#########################


LGLInterpTable and InterpFunction
#################################
.. _lgltab-guide:

The :code:`oc.LGLInterpTable` class inside of the optimal control module facilitates the interpolation of time-series data expressed in the
ODE format. It is distinct from :code:`InterpTable1D` discussed in :ref:`1-D Interpolation`. 
This can be used for reintegration of converged trajectories and can also be used incorporate
arbitrary time-series data into a vector function expression. To illustrate its construction and usage, we will utilize the
simple two-body low-thrust ODE below.


.. code-block:: python
	
    import asset_asrl as ast
    import matplotlib.pyplot as plt
    import numpy as np


    vf = ast.VectorFunctions
    Args = vf.Arguments
    oc = ast.OptimalControl

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

To show how we can construct an :code:`oc.LGLInterpTable` we will generate a trajectory by integrating our ODE with a control law.
This full ODE format trajectory can be provided along with the ODE, and its dimensions (state,control, and parameters), to the constructor. The :code:`ode` function
will be used to generate the derivatives of a cubic-hermite spline representation of the trajectory. Alternatively, we can construct an :code:`oc.LGLInterpTable`
from arbitrary time series data (not necessarily a full ODE trajectory). For example, below we generate another table to interpolate just the controls
from our integrated trajectory. Note that time is assumed to be the last element in each element in the list of data. You may then interpolate
values by supplying a time to the call operator of the object.

.. code-block:: python

    ode = TwoBodyLTODE(1,.01)

    r  = 1.0
    v  = 1.1
    t0 = 0.0
    tf = 20.0

    X0t0U0 = np.zeros((10))
    X0t0U0[0]=r
    X0t0U0[4]=v
    X0t0U0[6]=t0        

    def ULaw(throttle):
        V = Args(3)
        return V.normalized()*throttle

    integULaw   = ode.integrator("DP54",.1,ULaw(0.8),[3,4,5])

    TrajI   = integULaw.integrate_dense(X0t0U0,tf,6000)


    # Construct from an ode,its,dimensions,and a trajectory of the correct size
    # Most accurate interpolation
    Tab1 = oc.LGLInterpTable(ode.vf(),6,3,TrajI)


    ## Construct from arbitrary time series data,
    ## Elements consist of data followed by time
    ## No ode needed, but less accurate interpolation
    JustUts = [ list(T[7:10]) +[T[6]] for T in TrajI  ]
    Tab2 = oc.LGLInterpTable(JustUts)


    # Interpolation returns all data stored in the table, including the time

    print(Tab1(0.0)) # prints [1.,  0.,  0.,  0.,  1.1, 0.,  0.,  0.,  0.8, 0. ]

    print(Tab2(0.0)) # prints [0.,  0.8, 0.,  0. ]

:code:`oc.LGLInterpTable` objects may be supplied to the constructor of an integrator, in which case they are interpreted
as a time dependent control law. If the table contains data of the same size as the ODE's input, the correct
control indices are automatically calculated. However, if the data dimensions are not consistent with the ODE's input size, 
you must specify which elements of the interpolated output are to be interpreted as controls.
	
.. code-block:: python
    
    # Tables consisting of full trajectories of the right size will be interpreted 
    # use the control indices as a time dependent control law ([7,8,9])
    integTab1 = ode.integrator(.1,Tab1)

    ## If the data is not the same size as an ODE input you should manually
    ## Specify which elements of the outputs of the table should be controls
    ## Since Tab1 is the right size, this does the same thing as above
    integTab1 = ode.integrator(.1,Tab1,range(7,10))

    # However, Tab2 is just controls so we need to specify which elements of 
    # the output of the table are the controls 
    integTab2 = ode.integrator(.1,Tab2,range(0,3))

    Traj1   = integTab1.integrate_dense(X0t0U0,tf)
    Traj2   = integTab2.integrate_dense(X0t0U0,tf)

You can also utilize :code:`oc.LGLInterpTable` objects inside of VectorFunction expressions by wrapping them
with :code:`oc.InterpFunction` as shown below. To do this we pass a table instance as well as a list of the indices of the outputs
of the table that we want be included as outputs of our VectorFunction. :code:`oc.InterpFunction` objects can be quite useful
in representing time dependent boundary constraints that are not analytic functions of time.

.. code-block:: python
    
    # A constraint that enforces that given state and time should
    # match that held in the table
    def RendFunc(Tab):
        X,t = Args(7).tolist([(0,6),(6,1)])
    
        # Convert table into a vector function
        # that takes a time and returns the specified elements in the table
        # in this case just, position and velocity
        X_tfunc = oc.InterpFunction(Tab,range(0,6))
    
        return X-X_tfunc(t)
    
    RFunc = RendFunc(Tab1)

    
    print(RFunc(TrajI[-1][0:7]))  # prints [0,0,0,0,0,0]

