==============================
Optimal Control Phase Tutorial
==============================

Like many other optimal control packages, ASSET divides a large, potentially heterogeneous optimal control
problem into distinct phases. In ASSET, the dynamics along a phase of the trajectory are governed by one ODE and
discretized using a single transcription scheme. We may apply custom constraints and objectives to phases and optimize them by themselves, or
combine multiple phases and optimize them all simultaneously. 

To construct a phase we must first define an ODE using what we have learned in previos sections. In this section
we will utilize the trivial ODE below as a reference while we discuss the phase API.


.. code-block:: python

    class DummyODE(oc.ODEBase):
        def __init__(self,xv,uv,pv):
            args = oc.ODEArguments(xv,uv,pv)
            super().__init__(args.XVec(),xv,uv,pv)
        
    
    




Initialization
==============

Given some ASSET ode,we can now go about constructing the phase object
using its .phase. At minimum, we must first specify the transcription mode for the phase dynamics as a string. 
Here we have chosen, the  third order Legendre gauss lobatto collocation or LGL3, which approximates the trajectory as piecewise cubic splines. We can also
choose from the 5th and 7th Order LGL methods, the trapezoidal method, or a central shooting scheme. In most
cases we suggest first trying the LGL3 scheme, however 5th and 7th methods may be superior for some applications.
Additionally, users should prefer the LGL collocation methods over the central shooting scheme
for almost all applications, as they are almost always significantly faster and more robust. 

.. code-block:: python

    ode = DummyODE(6,3,1)
    phase = ode.phase("LGL3")

We can then supply an initial guess, consisting of python list of full-states of the correct size,
using the setTraj method. In most cases we will just pass in the initial guess
and specify the number of segments of the chosen transcription type we want to
use to approximate the dynamics. By default these will be evenly spaced in time.
Note that the number of segments does not have to match the number
of states in the initial guess, nor do the states in the initial guess have to be evenly spaced.
We can also manually specify the initial spacing for segments. This is done by passing a python list
SegBinSpacing of length >=2 specifying the spacing on the non-dimensional time interval 0-1
for groups of evenly spaced segments. We then pass another list of length n-1 specifying
the number of segments we want in each group. For example, we can replicate the behavior of the
default method as shown below. Or alternatively, we could specify that we want to vary the density of 
segments across the phase. In most cases, users first option should be to just evenly space segments over
the phase. One can also create a phase, set the transcription method, and initial guess in the same
call as shown on the final line. 

.. code-block:: python

    ## 500 Segments evenly spaced over entire time interval
    phase.setTraj(InitialGuess,500)

    ## 500 Segments evenly spaced over entire time interval
    SegBinSpacing = [0.0 , 1.0]
    SegsPerBin=        [500]
    phase.setTraj(InitialGuess,SegBinSpacing,SegsPerBin)

    # 300 segments spaced over first half of trajectory, 200 over last half
    SegBinSpacing = [0.0 , 0.5 , 1.0]
    SegsPerBin       =[ 300 , 200]
    phase.setTraj(InitialGuess,SegBinSpacing,SegsPerBin)

    ## create phase, set Transcription, IG, and number of segments
    phase = ode.phase("LGL3",InitialGuess,500)

In addition to specifying the transcription mode, we can also choose from several
different control parameterizations, using the .setControlMode method. 

.. code-block:: python

    # Options: FirstOrderSpline,HighestOrderSpline,BlockConstant,NoSpline
    phase.setControlMode("FirstOrderSpline")

By default,this is set to 'FirstOrderSpline', which will ensure that the control history has smooth
first derivatives (if possible for the chosen transcription). This typically sufficient to prevent control chattering in
the LGL5, and LGL7 methods. We can also set the ControlMode to HighestOrderSpline to enforce continuity
in all derivatives possible for a given transcription method. For LGL5, the control is represented as piecewise
quadratic function, so FirstOrderSPline and HighestOrderSpline are equivalent. For LGL7, the control is represented
a piecewise cubic function, therefore setting control mode to "HighestOrderSpline" will ensures that this cubic function
has smooth first and second derivatives. For the 'LGL3','Trapezoidal',and 'CentralShooting' schemes, the control
history is piecewise linear across segments and does need any regularization, thus for those methods, FirstOrderSpline and 
HighestOrderSpline have no effect.

Alternatively, For all methods, we can also specify that rather than having a smooth control history, we want to have a piecewise
constant control history with 1 unique control per segment. This can be specified by setting the control mode to 'BlockConstant'.
In our experience this control parameterization can be very robust and typically results in KKT matrices that are faster to factor.
The caveat is that special care must be taken when reintegrating converged solutions with an explicit integrator. This will be covered in a later section. 

In addition to the state time and ODE parameter variables representing the trajectory, we may also add what we call Static Parameters to the phase. These are non-time varying
variables that you might need to formulate a custom constraint and objective that are not needed by the dynamics. Note that these are not the same as ODEParamters.
We can add static parameters by simply specifying their initial values as shown below.

.. code-block:: python

    phase.setStaticParams([0.0,0.0])




.. list-table:: List of Phase Transcription Options
   :widths: 15 25 20 40
   :header-rows: 1

   * - Name
     - Description
     - Integral Method
     - Control Representation
   * - :code:`'LGL3'`
     - Third order Legendre Gauss Lobatto collocation.
       Two states per segment.
     - Trapezoidal Rule
     - Piecewise-Linear ('NoSpline'), Piecewise-Constant ('BlockConstant') 
   * - :code:`'LGL5'`
     - Fifth order Legendre Gauss Lobatto collocation.
       Three states per segment.
     - Simpson's Rule
     - Quadratic-Spline ('FirstOrderSpline'), Piecewise-Constant ('BlockConstant')
   * - :code:`'LGL7'`
     - Seventh order Legendre Gauss Lobatto collocation.
       Four states per segment.
     - Unnamed fourth order quadrature method
     - Cubic-Spline ('FirstOrderSpline'),
       Natural-Cubic-Spline ('HighestOrderSpline'),
       Piecewise-Constant ('BlockConstant')
   * - :code:`'Trapezoidal'`
     - Trapezoidal collocation.
       Two states per segment.
     - Trapezoidal Rule
     - Piecewise-Linear ('NoSpline'), Piecewise-Constant ('BlockConstant')
   * - :code:`'CentralShooting'`
     - Adaptive Dormand Prince 8(7) central shooting method.
       Two states per segment.
     - Trapezoidal Rule
     - Piecewise-Linear ('NoSpline'), Piecewise-Constant ('BlockConstant')

Constraints and Objectives
=========================

Before discussing the interface for adding different types of constraints, it is helpful to briefly overview how we represent a phases's variables
when formulating an optimization problem. In general we portion a trajectory with n states into each time-varying portion Vi of each full-state followed by the
ODEParams and Static Params below. 

.. math::
   :name: eq:1


   \vec{x} = \begin{bmatrix}
              \vec{V}_1     \\
              \vec{V}_2     \\
              \vdots        \\
              \vec{V}_{n-1} \\
              \vec{V_n}     \\
              \vec{P}       \\
              \vec{S}       \\
             \end{bmatrix}
    \quad \quad \text{where} \quad \vec{V}_i = [\vec{X}_i,t_i,\vec{U}_i]

The transcription defect constraints, and segment mesh spacing constraints are formulated automatically by the phase object from these variables, and users should not
attempt to formulate them on their own. Every other constraint and objective must be specified by the user, in terms this discrete representation of the trajectory. To simplify this process,
and provide an interface that is invariant to the number of segments, phase only allows you to write constraints/objective that gather inputs form certain "Phase Regions" in the total variables
vector. A complete list of the currently allowed phase regions is listed below and we will discuss how you can use them in the next section.


.. list-table:: Phase Regions
   :widths: 15 50 35
   :header-rows: 1

   * - Phase Region
     - Description
     - Input Order
   * - Front
     - Applied to first time-varying-state, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_0,\vec{P},\vec{S}])`
   * - Back
     - Applied to last time-varying-state, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_n,\vec{P},\vec{S}])`
   * - Path
     - Applied to every time-varying-state, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{P},\vec{S}]),\; i = 1\ldots n`
   * - InnerPath
     - Applied to every time-varying-state (excluding the first and last), the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{P},\vec{S}]),\; i = 2\ldots n-1`
   * - FrontandBack
     - Applied to the first and last time-varying-states, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_1,\vec{V}_n,\vec{P},\vec{S}])`
   * - PairwisePath
     - Applied to every pair of adjacent time-varying-states, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{V}_{i+1},\vec{P},\vec{S}]),\; i = 1\ldots n-1`
   * - ODEParams
     - Applied to the ODE parameters.
     - :math:`\vec{f}([\vec{P}])`
   * - StaticParams
     - Applied to the phase's static parameters.
     - :math:`\vec{f}([\vec{S}])`


Equality Constraints
--------------------
Equality constraints of the form h(\vec{x}) = 0, can be added to a phase using the .addEqualConMethod(). First we spenify 
the phase region, specifies from which time-varying state to which the constraint will be applied followed by the equality 
constraint itself (an ASSET VEctor function). Next which of the indices of time-varying state variables at the phaseregion,
as well as any ODEParams and phase Static Parameters we wish to forward to the function. In the trivial example below, we are adding 
constraint that enforces that the first-time-varying state in the trajectory and all of the ODE parameters should be equal to zero. 
However, our constraint must be written that such that the inputs consist of the time-varying states, followed by the ODE parameters(if any), and static
parameters (if any).

.. code-block:: python

    PhaseRegion = "First"

    def AnEqualCon():
        XtU_OP_SP = Args(13)
        return XtU_OP_SP

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function
    
    phase.addEqualCon(PhaseRegion,AnEqualCon(),XtUVars,OPVars,SPVars)

It should be noted that we do not have to include every variable in a phase region for every constraint. Instead, they may and should be written in terms of 
only the variables they actually need. For example, below, we add a constraint involving the second and third state variables from the 
last phase in the trajectory, as well as the first ODE parameter and second static parameter.

.. code-block:: python

    PhaseRegion = "Last"

    ## Only need second and third state variables, the first ode parameter, and the second static parameter
    def AnotherEqualCon():
        x1,x2,op0,sp1 = Args(4).tolist()
        return vf.sum(x1,x2,op0/sp1) + 42.0

    XtUVars = [1,2]  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = [0]    # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = [1]    # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function

    phase.addEqualCon(PhaseRegion,AnotherEqualCon(),XtUVars,OPVars,SPVars)

Furthermore, when variables from only a single grouping are needed we do not have to pass them as an argument, as illustrated in the three
examples below.

.. code-block:: python

    XtUVars = [7,8,9]  # Just the controls and nothing else
    # enforce unit norm of all control vectors
    phase.addEqualCon("Path",Args(3).norm()-1.0,XtUVars,[],[]) 
    # same as above
    phase.addEqualCon("Path",Args(3).norm()-1.0,XtUVars) 


    OPVars = [0]  # Just the ODEParam
    #Enforce Square of first ODE param = 4
    phase.addEqualCon("ODEParams",Args(1)[0]**2 - 4.0,[],OPVars,[]) 
    # same as above
    phase.addEqualCon("ODEParams",Args(1)[0]**2 - 4.0,OPVars) 


    SPVars = [0,1]  # Just the static params
    #Enforce sum of static params = 2
    phase.addEqualCon("StaticParams",Args(2).sum() - 2.0,[],[],SPVars) 
    # same as above
    phase.addEqualCon("StaticParams",Args(2).sum() - 2.0,SPVars) 

The previous examples, only illustrate the usage of the phase regions that take at most one state, however, phase regions "FrontandBack", and
"PairwisePath" take up to two states. An example of how to use a two state phase region is shown below. Here we are constraining that the first and last states should be equal
and that the difference between the last and first time of the phase should be equal to a static parameter that we have added to the phase. We only specify that which time-varying state varaibels
we want once. The same set is gathered from the First state and last state and forwarded to the function, followed by any ODE parameters (none in this case) 
and statics parameters.

.. code-block:: python

    def FrontBackEqCon():
        X_0,t_0,X_f,t_f,sp0 = Args(15).tolist([(0,6),(6,1),(7,6),(13,1),(14,1)])
    
        eq1 = X_0-X_f
        eq2 = t_f-t_0 - sp0
        return vf.stack(eq1,eq2)
     

    XtUVars = range(0,7)  # index of time
    SPVars  = [0]  # first static parameter
    # Constrain first and last states to be equal and
    # constrain Delta Time over the phase (tf-t0) to be equal to the first static parameter
    phase.addEqualCon("FirstandLast",FrontBackEqCon(),XtUVars,[],SPVars)

In addition to the general methods of adding equality constraints illustrated in the previous examples, there are several additional methods to simplify the 
definition of commonly occurring types of constraints. By far the most commonly used is the addBoundaryValue method, which simply adds a constraint that the specified
variables should be equal to some vector (numpy vector or python list) constants. This method is typically used to 
enforce known initial and terminal conditions on a phase.

.. code-block:: python
    
    XtUVars = [1,3,9]
    Values  = [np.pi,np.e,42.0]
    phase.addBoundaryValue("First",XtUVars,Values)

    OPVars  = [0]
    Values  = [10.034]
    phase.addBoundaryValue("ODEParams",OPVars,Values)

    SPVars = [0,1]
    Values  = [1.0,4.0]
    phase.addBoundaryValue("StaticParams",SPVars,Values)


Additionally, we also provide methods to constrain changes in variables from the first to last variables to a specified value. This could for example
be used to enforce a fixed duration for the phase.

.. code-block:: python

    phase.addDeltaVarEqualCon(0,1.0)
    # These do the same as the following
    DeltaEqualCon= Args(2)[1]-Args(2)[0] -1.0
    phase.addEqualCon("FirstandLast",DeltaEqualCon,[0])

    ## These do the same thing, constraining the elapsed time over the phase to be = 1.0
    phase.addDeltaVarEqualCon(6,1.0)
    phase.addDeltaTimeEqualCon(1.0) #Time is special and has its on named method

    # Both are equivalent to the following
    DeltaEqualCon= Args(2)[1]-Args(2)[0] -1.0
    phase.addEqualCon("FirstandLast",DeltaEqualCon,[6])





Inequality Constraints
----------------------
Adding  general inequality constraints, using phase.addInequalCon, works exactly the same as it did for addEqualCon. The only difference
is that our functions should be constraints should be of the form f(x)<0. In other words, we assume that out function is in the feasible region whenever
its value is negative. For example, if we wanted to add a constraint specifying that all of the initial time-varying state variables, ODE Parameters, and Static
parameters should be positive we could implement that as shown below.

.. code-block:: python

    PhaseRegion = "First"

    def AnInequalCon():
        XtU_OP_SP = Args(13)
        return -1.0*XtU_OP_SP

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function
    
    phase.addInequalCon(PhaseRegion,AnInequalCon(),XtUVars,OPVars,SPVars)

    # Same rules as covered for addEqualCon
    phase.addInequalCon("Path", Args(4).sum(),[0,1,2],[],[1])
    phase.addInequalCon("Back",  Args(3).squared_norm()-1,[3,4,5])
    phase.addInequalCon("StaticParams",1-Args(2).norm(),[0,1])


However, it is somewhat cumbersome to write many of the types of inequality constraints that you will encounter using this method, thus
we offer many simplified options which we now discuss.

The simplest type of inequality constraint we can apply are simple bounds on the variables. These can be added using the addLower/Upper/LUVarBounds methods as 
shown below, these can be added to any of the single-time-varying state phase regions. For any methods we can also specify a positive scale factor that will be 
applied to the final bound. This can help scale an ill conditioned bound but will not change the meaning of the constraint.


.. code-block:: python
    
    # Add lower bound to the 7th state,time,control variable
    PhaseRegion = "Back"
    VarIndex    = 7
    LowerBound  = 0.0
    Scale       = 1.0  # strictly positive scale factor

    phase.addLowerVarBound(PhaseRegion,VarIndex,LowerBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLowerVarBound(PhaseRegion,VarIndex,LowerBound)


    # Add upper bound to the 7th state,time,control variable
    PhaseRegion = "Back"
    VarIndex    = 7
    UpperBound  = 1.0
    Scale       = 1.0  # strictly positive scale factor

    phase.addUpperVarBound(PhaseRegion,VarIndex,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addUpperVarBound(PhaseRegion,VarIndex,UpperBound)


    ## Add Both Lower and Upper Bounds at same time
    PhaseRegion = "Back"
    VarIndex    = 7
    LowerBound  = 0.0
    UpperBound  = 1.0
    Scale       = 1.0  # strictly positive scale factor for both bounds

    phase.addLUVarBound(PhaseRegion,VarIndex,LowerBound,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLUVarBound(PhaseRegion,VarIndex,LowerBound,UpperBound)

    # Also works for the parameter variables
    phase.addLUVarBound("StaticParams",0,-1.0,1.0)

    # Violations are now of order one
    Scale = 10000.0
    phase.addUpperVarBound("ODEParams",0,1.0/10000.0, Scale)

In addition to placing bounds on variables, you can also place bounds on the outputs of scalar functions of variables. This
is accomplished using the addLower/Upper/LUFuncBounds methods as shown below. In this example we are showing various ways to bound norm
of all of the controls (variables [7,8,9]) to be between 0 and 1.0.

.. code-block:: python

    ## Upper bound on the norm of the controls
    PhaseRegion ="Path"
    ScalarFunc = Args(3).norm()
    XTUVars = [7,8,9]
    UpperBound = 1.0
    Scale = 1.0

    phase.addUpperFuncBound(PhaseRegion,ScalarFunc,XTUVars,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addUpperFuncBound(PhaseRegion,ScalarFunc,XTUVars,UpperBound)


    ## Lower bound on the norm of the controls
    PhaseRegion ="Path"
    ScalarFunc = Args(3).norm()
    XTUVars = [7,8,9]
    LowerBound = 0.0
    Scale = 1.0

    phase.addLowerFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLowerFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound)


    ## Lower and Upper on squared norm at the same time
    PhaseRegion ="Path"
    ScalarFunc = Args(3).squared_norm()
    XTUVars = [7,8,9]
    LowerBound = 0.0
    UpperBound = 1.0

    Scale = 1.0

    phase.addLUFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLUFuncBound(PhaseRegion,ScalarFunc,XTUVars,LowerBound,UpperBound)


These function, bounding methods can be applied to any scalar function you wish to bound, however, the examples above
of bound the norm or squared norm are so common that we also provide methods that do just that. Below, we use the addLower/Upper/NormBound
and addLower/Upper/SqauredNormBound methods the accomplish the  same tasks as the previous code block.

.. code-block:: python
    
    ## Upper bound on the norm of the controls
    PhaseRegion ="Path"
    XTUVars = [7,8,9]
    UpperBound = 1.0
    Scale = 1.0

    phase.addUpperNormBound(PhaseRegion,XTUVars,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addUpperNormBound(PhaseRegion,XTUVars,UpperBound)


    ## Lower bound on the norm of the controls
    PhaseRegion ="Path"
    XTUVars = [7,8,9]
    LowerBound = 0.0
    Scale = 1.0

    phase.addLowerNormBound(PhaseRegion,XTUVars,LowerBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLowerNormBound(PhaseRegion,XTUVars,LowerBound)


    ## Both at the same time
    PhaseRegion ="Path"
    XTUVars = [7,8,9]
    LowerBound = 0.0
    UpperBound = 1.0

    Scale = 1.0

    phase.addLUSquaredNormBound(PhaseRegion,XTUVars,LowerBound,UpperBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLUSquaredNormBound(PhaseRegion,XTUVars,LowerBound,UpperBound)


Similar to how, we showed how to place equality constraints on the change in a variable from beginning to the end of
a phase, we can also place bounds on the changes in variables as shown below.

.. code-block:: python

    VarIdx     = 0
    LowerBound = 0.0
    Scale      = 1.0

    phase.addLowerDeltaVarBound(VarIdx,LowerBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addLowerDeltaVarBound(6,LowerBound)


    VarIdx     = 0
    UpperBound = 1.0
    Scale      = 1.0


    phase.addUpperDeltaVarBound(VarIdx,LowerBound,Scale)
    # If no scale factor is supplied it is assumed to be = 1.0
    phase.addUpperDeltaVarBound(VarIdx,LowerBound)


    # Time is special, we can use addLower/UpperDeltaTimeBound instead
    LowerBound = .5
    UpperBound = 1.5
    Scale      = 1.0

    phase.addLowerDeltaTimeBound(LowerBound,Scale)
    phase.addUpperDeltaTimeBound(UpperBound)
    


State Objectives
----------------
The simplest type of objective function that we can add to phase is a state objective. It is a scalar function that we
wish to directly minimize that takes some or all of the variables at a phase region. Note, then that if you are trying to maximize something you should
multiply its value by -1.0. Generalized state objectives can be added to a phase using the addStateObjective function as shown below. The same rules governing
addEqualCon, and addEqualCon apply here to all possible permutations to the inputs of addStateObjective. The only exception being that the function
must be an asset scalar function.

.. code-block:: python

    def AStateObjective():
    XtU_OP_SP = Args(13)
    return XtU_OP_SP.norm()  ## An Asset Scalar Function

    PhaseRegion = "Back"
    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function

    phase.addStateObjective(PhaseRegion,AStateObjective(),XtUVars,OPVars,SPVars)

In addition to the general methods, we also provide two more specialized methods that encompass the most common types of state objectives.

The first is the addValueObjective method which simply adds an objective function specifying that we want to minimize the value of one the variables
at a specified phase region multiplied by a scalar factor. To maximize the value, make the scale factor negative.

.. code-block:: python

    # Minimize the final value of XtUVar 5 
    PhaseRegion = "Last"
    VarIdx = 5
    Scale = 1.0
    phase.addValueObjective(PhaseRegion,VarIdx,Scale)


    # Maximize the initial value of XtUVar 0 
    PhaseRegion = "First"
    VarIdx = 0
    Scale = -1.0  ## Negative scale factors to maximize!!!
    phase.addValueObjective(PhaseRegion,VarIdx,Scale)


    # Minimize the Static Param 0 
    PhaseRegion = "StaticParams"
    VarIdx = 0
    Scale = 1.0
    phase.addValueObjective(PhaseRegion,VarIdx,Scale)

The second is the  addDeltaVarObjective which adds an objective to minimize the change in the value of some variable across the phase multiplied by 
a scale factor. AS before, to maximize the change, make the scale factor negative.

.. code-block:: python

    # Minimize change in XtUVar 2 across the phase ie: x2_f - x2_0
    VarIdx = 2
    Scale  = 1.0
    phase.addDeltaVarObjective(VarIdx,Scale)


    # Maximize change in XtUVar 4 across the phase ie: x4_f - x4_0
    VarIdx = 4
    Scale  = -100.0  # Negative scale factor to maximize
    phase.addDeltaVarObjective(VarIdx,Scale)

    # Minimize the duration of the phase : tf-t0
    VarIdx = 6  # Index of time
    Scale  = 1.0
    phase.addDeltaVarObjective(VarIdx,Scale)
    ## Time is special and has its own named method that does the same as above
    phase.addDeltaTimeObjective(Scale)




Integral Objectives
-------------------
The other common type of objective function we can add to phase are integral objectives. To add an integral objective, 
we provide a scalar integrand function to the phase using the addIntegralObjective method. 
The quadrature method used to approximate the integral will be depend on the current transcription type and are given in table Z.
When adding integral objectives as shown below,we only need to provide the integrand function and the
indices from the various variable groupings we want to forward to the integrand. 

.. code-block:: python

    def AnIntegrand():
        XtU_OP_SP = Args(13)
        return XtU_OP_SP.norm()  ## An Asset Scalar Function

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indcices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indcices of the phase Static Parameters (indexed from 0) we want to forward to our function


    # Signature if variables of all types are needed by integrand
    phase.addIntegralObjective(AnIntegrand(),XtUVars,OPVars,SPVars)

    # Signature if only state,time, and control variables needed by integrand
    phase.addIntegralObjective(Args(3).norm(),[7,8,9])

    # All integrands are minimized, so to maximize, multiply by negative number
    phase.addIntegralObjective(-10.0*Args(4).norm(),[1,2,3,7])
    


Integral Parameter Functions
----------------------------

The final class of functions that we can add to a phase are what we call Integral Parameter functions, and are used along with static parameters
to facilitate integral constraints on a phase. An integral parameter function is a special equality constraint of the form. 

.. math::
   
   \int_{t_0}^{t_f} f([\vec{X},t,\vec{U},\vec{P},\vec{S}_{\not k}]) dt - s_k = 0

Essentially, this constraint will force the value of one the static parameters to be equal to the integral of a user specified function. 
One can then place constraints on this static parameter using any of the previously discussed methods.
Adding in an integral parameter function, using addIntegralParamFunction works essentially the same as addIntegralObjective, except we also provide the index
of the static parameter the integral value will be assigned to.

.. code-block:: python

    def AnIntegrand():
        XtU_OP_SP = Args(12)
        return XtU_OP_SP.norm()  ## An Asset Scalar Function

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,1)   # indices of the phase Static Parameters (indexed from 0), MINUS THE ONE WE ARE ASSIGING THE INTEGRAL TOO
    IntSPVar = 1 # Assign the value of the integral to the second static parameter

    # Signature if variables of all types are needed by integrand
    phase.addIntegralParamFunction(AnIntegrand(),XtUVars,OPVars,SPVars,IntSPVar)

    # Signature if only state,time, and control variables needed by integrand
    phase.addIntegralParamFunction(Args(3).norm(),[7,8,9],IntSPVar)

    ## Now we can apply constraints to the integral by constraining the static param
    # Ex: constrain the integral to be equal to 100.0
    phase.addBoundaryValue("StaticParams",[1],[100.0])

    

Solving and Optimizing
======================

After formulating a phase by supplying an initial guess, and adding constraints/objectives, we can now use psiopt to solve or optimize
the trajectory. The settings of the optimizer can be manipulated through a reference to PSIOPT attached to the phase object. However, calls to
the optimizer are handled through the phase itself as shown below. Both of these topics are handled in more details in the section on PSIOPT.


.. code-block:: python

    phase.optimizer ## reference to this phases instance of psiopt
    phase.optimizer.set_OptLSMode("L1")

    
    ## Solve just the dynamics,equality, and inequality constraints
    flag = phase.solve()

    ## Optimize objective subject to the dynamic,equality, and inequality constraints
    flag = phase.optimize()

    ## Call solve to find feasible point, then optimize objective subject to the dynamic,equality, and inequality constraints
    flag = phase.solve_optimize()

    ## Same as above but calls solve if the optimize call fails to fully converge
    flag = phase.solve_optimize_solve()


After finding a solution, to the phase, we can retrieve the converged trajectory using the returnTraj method of the phase. Similarly,
if you added static parameters to the phase, these can be retrieved using returnStaticParams. Finally, you can also retrieve an estimate for the Co-states
of a solved optimal control problem. These could then be used as the initial guess to an indirect form of the same optimization problem.

.. code-block:: python
    
    Traj = phase.returnTraj()

    StatParams = phase.returnStaticParams()

    CostateTraj = phase.returnCostateTraj() #

Additionally, should you want to refine the mesh spacing of the trajectory after a solution, it is not necessary to create an entirely new phase.
Instead we can use the refineTraj methods as shown below. The simplest form of refine is accomplished using the refineTrajManual. In general these work exactly,
the same as the setTraj Method except they use the currently loaded trajectory to interpolate the new mesh. The second, option is the refineTrajEqual method,
which will attempt to re-mesh the trajectory such the estimated error across all segments is equal. For now, users should prefer manual mesh refinement, as our
automatic refinement method could use some serious improvement.

.. code-block:: python

    phase.optimize() # optimize or solve initial mesh
  

    phase.refineTrajManual(1000) # remesh trajectory with 1000 evenly spaced segments
    phase.optimize() # optimize or solve new mesh

    ## Manually Specify spacing
    # 600 segments spaced over first half of trajectory, 400 over last half
    SegBinSpacing = [0.0 , 0.5 , 1.0]
    SegsPerBin       =[ 600 , 400]
    phase.refineTrajManual(SegBinSpacing,SegsPerBin)
    phase.optimize() # optimize or solve new mesh


    ## Remesh with 1000 segments spaced to have approximately equal error per segment
    phase.refineTrajEqual(1000)
    phase.optimize() # optimize or solve new mesh

    TrajRef = phase.returnTraj()




Solving and Optimizing
======================




