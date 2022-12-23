.. _ode-guide:

ODE Tutorial
============

One of ASSET's primary goals is facilitating optimal control and integration of dynamical systems governed by ordinary differential equations (ODEs).
In ASSET, an ODE is simply a VectorFunction adhering to the state-space formalism shown below. 
That is, it is a VectorFunction that takes as arguments the state of the system, :math:`\vec{X}`, the current time,
:math:`t`, time varying controls, :math:`\vec{U}`, some static ODE parameters :math:`\vec{P}`, and returns the time derivative of the state variables. 

.. math::

    \begin{equation}
    \dot{\vec{X}} = \vec{F}([\vec{X},t,\vec{U},\vec{P}])
    \end{equation}


Some ODEs do not require the :math:`\vec{U}` or :math:`\vec{P}` inputs (e.g. ballistic gravity model), and thus it is not necessary to provide them in those cases. 
However, because we use a common interface between autonomous and non-autonomous ODEs, it is always necessary to provide 
an explicit time variable even if it is unused in the dynamics. 


ODEs can be written using any of the techniques described in the VectorFunctions
:ref:`section <vectorfunction-guide>`, provided that they obey the state space formalism for their inputs and outputs. 
For example, a simple two body gravitational model with low thrust could be written as shown 
below. This model would have 6 state variables representing the position and velocity vectors
relative to the central body, and three control variables representing the thrust direction and engine throttle.
We implicitly assume that we will later place an upper bound of 1 on the norm of the control so that the acceleration
never exceeds the maximum we specify here. Adhering to the convention above, the position and velocity state variables will be
the first 6 input arguments, followed by the time, and then the thrust direction/throttle. As outputs we return
the time derivatives of position and velocity in the same order we assumed in the input arguments.


.. code-block:: python

    import asset_asrl as ast
    import numpy as np

    vf        = ast.VectorFunctions
    oc        = ast.OptimalControl
    Args      = vf.Arguments

    def TwoBodyLTFunc(mu,MaxLTAcc):
    
        XtU = Args(10) # [ [R,V] , t, [U]]
    
        R,V,t,U  = XtU.XVec().tolist([(0,3),(3,3),(6,1),(7,3)])
    
        G = -mu*R.normalized_power3()
    
        Acc = G + U*MaxLTAcc
    
        Rdot = V
        Vdot = Acc
    
        ode = vf.stack([Rdot,Vdot])
        return ode

ODEArguments
############

To simplify the process of defining VectorFunctions adhering to the state space
formalism, we provide the :code:`ODEArguments` class inside of the :code:`OptimalControl` module.
This class is a thin wrapper around the :code:`Arguments` class that allows you to index relevant
sub-vectors and elements of an ODE's inputs in a clearer way than using :code:`Arguments`. To construct 
:code:`ODEArguments` we pass the number of state variables, control variables (if any), and ODE parameters (if
any). The total input size will be the sum of :code:`XVars`, :code:`PVars`, and :code:`UVars` plus 1 for time. We can then
address the relevant sub-vectors of our input using the :code:`X/U/PVec()` methods. These methods are return regular
segment types so we can then apply all operations we would to those objects. Similarly, we can also address
specific elements of each of these sub-vectors using the :code:`X/U/PVar(i)` methods.

.. code-block:: python

    def TwoBodyLTFunc(mu,MaxLTAcc):
    
        XVars = 6
        UVars = 3
        PVars = 0
    
        XtU = oc.ODEArguments(XVars,UVars,PVars) # [ [R,V] , t, [U],[]]
    
        # no need to specify PVars if there aren't any, same would go for UVars
        # if there were no control variables
        XtU = oc.ODEArguments(XVars,UVars) 
        
        # Index state,control or parameter vectors
        XVec = XtU.head(6)
        XVec = XtU.XVec()
    
        UVec = XtU.segment(XVars+1,UVars)
        UVec = XtU.UVec()
    
        r0,r1,r2,v0,v1,v2 = XVec.tolist()
        u0,u1,u2 = UVec.tolist()
    
        # If we had ode parameters
        #PVec = XtU.segment(XVars+1+UVars,PVars)
        #PVec = XtU.PVec()
    
        R,V  = XVec.tolist([(0,3),(3,3)])
        U = UVec
    
        ### Index specific elements 
        t  = XtU.TVar()   # same as XtU[XVars]
    
        v1 = V[1]
        v1 = XVec[4]
        v1 = XtU.XVar(4) # XtU.UVar(i) is same as XtU[i]
    
        u0 = UVec[0]
        u0 = XtU.UVar(0) # XtU.UVar(i) is same as XtU[XVars+i]
    
        ######################################
        G = -mu*R.normalized_power3()
        Acc = G + U*MaxLTAcc
    
        Rdot = V
        Vdot = Acc
    
        ode = vf.stack([Rdot,Vdot])
        return ode




Defining ODE Classes
####################

If you were to inspect the type of the result of the function above, it would
be :code:`VectorFunction`, and at this point ASSET has no idea that it is an ODE.
For ASSET to recognize our function as an ODE and allow us to use it directly with
all associated utilities, we need to define it using the class
based style described in :ref:`the vector functions tutorial <vfstyle-guide>`, but inherit from the class :code:`oc.ODEBase`
rather than :code:`VectorFunction`. Therefore, the correct way to write the :code:`TwoBodyLT` ODE is shown below.
When initializing our base class we simply supply the asset VectorFunction specifying
the ode as well as the number of states, controls, and parameters.

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
        


        
    ode = TwoBodyLTODE(1,.01)

    phase = ode.phase("LGL3")
    integ = ode.integrator("DOPRI87",.1)

    

This object is now a full fledged ODE, from which we can dispatch phase and integrator
objects. We will discuss usage of these in the next sections.