.. _phase-guide:

==============================
Optimal Control Phase Tutorial
==============================

Like many other optimal control packages, ASSET divides a large, potentially heterogeneous optimal control
problem into distinct phases. In ASSET, the dynamics along a phase of the trajectory are governed by one ODE and
discretized using a single transcription scheme. We may apply custom constraints and objectives to phases and optimize them by themselves, or
combine multiple phases and optimize them all simultaneously. 

To construct a phase, we must first define an ODE using what 
we have learned in previous sections on :ref:`vector functions <vectorfunction-guide>` and :ref:`ODEs <ode-guide>`. In this section
we will utilize the trivial ODE below as a reference while we discuss the phase API.


.. code-block:: python

    class DummyODE(oc.ODEBase):
        def __init__(self,xv,uv,pv):
            args = oc.ODEArguments(xv,uv,pv)
            super().__init__(args.XVec(),xv,uv,pv)
        
    
    




Initialization
==============

Given some ASSET ODE, we can construct a phase object
using it's :code:`.phase` method. At minimum, we must first specify the transcription mode for the phase dynamics as a string. 
Here we have chosen, third order Legendre-Gauss-Lobatto collocation or :code:`'LGL3'`, which approximates the trajectory as piecewise cubic splines. We can also
choose from the 5th and 7th Order LGL methods, the trapezoidal method, or a central shooting scheme. In most
cases we suggest first trying the :code:`'LGL3'` scheme, however the 5th and 7th order methods may be superior for some applications.
Additionally, users should prefer the LGL collocation methods over the central shooting scheme
for almost all applications, as they are almost always significantly faster and more robust. 

.. code-block:: python

    ode = DummyODE(6,3,1)
    phase = ode.phase("LGL3")

We can supply an initial guess to the :code:`phase` using the :code:`.setTraj` method. 
The initial guess should be formatted as a python list where each element 
is a full ODE input (ie: :math:`[\vec{X}_i,t_i,\vec{U}_i,\vec{P}]`) at each point in time along the trajectory.
Note, that this is the same format as the output of an :code:`intgrate_dense` call for the ODE's integrator.
In most cases we will just pass in the initial guess
and specify the number of segments of the chosen transcription type we want to
use to discretize the dynamics. By default these will be interpolated from the initial guess to be evenly spaced in time.
Note that the number of segments does not have to match the number
of states in the initial guess, nor do the states in the initial guess have to be evenly spaced in time. However, you should
include enough states in the initial guess so that it can be re-interpolated with decent accuracy.

.. code-block:: python

    # the format of the input trajectory
    for XtUP in InitialGuess:
        XtUP[0:6]  # The state variables,X
        XtUP[6]    # The time,t
        XtUP[7:10] # The control variables,U
        XtUP[10]   # The ODE parameter,P

    ## 500 Segments evenly spaced over entire time interval
    phase.setTraj(InitialGuess,500)

We can also manually specify the initial spacing for segments. This is done by passing a python list
:code:`SegBinSpacing` of length :code:`n >=2` specifying the spacing on the non-dimensional time interval 0-1
for groups of evenly spaced segments. We then pass another list, :code:`SegsPerBin`, of length :code:`n-1` specifying
an integer number of segments we want in each group. For example, we can replicate the behavior of the
default method as shown below. Alternatively, we could specify that we want to vary the density of 
segments across the :code:`phase`. In most cases, the user's first option should be to just evenly space segments over
the phase. One can also create a :code:`phase`, set the transcription method, and initial guess in the same
call as shown on the final line. 

.. code-block:: python

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
different control parameterizations, using the :code:`.setControlMode` method. 

.. code-block:: python

    # Options: FirstOrderSpline,HighestOrderSpline,BlockConstant,NoSpline
    phase.setControlMode("FirstOrderSpline")

By default, this is set to :code:`'FirstOrderSpline'`, which will ensure that the control history has smooth
first derivatives (if possible for the chosen transcription). This is typically sufficient to prevent control chattering in
the :code:`'LGL5'`, and :code:`'LGL7'` methods. We can also set the ControlMode to :code:`'HighestOrderSpline'` to enforce continuity
in all derivatives possible for a given transcription method. For :code:`'LGL5'`, the control is represented as a piecewise
quadratic function, so :code:`'FirstOrderSpline'` and :code:`'HighestOrderSpline'` are equivalent. For :code:`'LGL7'`, the control is represented
as a piecewise cubic function, therefore setting control mode to :code:`'HighestOrderSpline'` will ensures that this cubic function
has smooth first and second derivatives. For the :code:`'LGL3'`, :code:`'Trapezoidal'`,and :code:`'CentralShooting'` schemes, the control
history is piecewise-linear across a segment and does need any regularization, thus for those methods, :code:`'FirstOrderSpline'` and 
:code:`'HighestOrderSpline'` have no effect.

Alternatively, for all methods, we can also specify that rather than having a smooth control history, we want to have a piecewise
constant control history with 1 unique control per segment. This can be specified by setting the control mode to :code:`'BlockConstant'`.
In our experience this control parameterization can be very robust and typically results in KKT matrices that are faster to factor.
The caveat is that special care must be taken when re-integrating converged solutions with an explicit integrator. This will be covered in a later section. 

In addition to the state, time, control, and ODE parameter variables representing the trajectory, we may also add what we call "static parameters" to the :code:`phase`. 
These are non-time varying variables that you might need to formulate a custom constraint and objective that are not needed by the dynamics. 
Note that these are not the same as ODE parameters. We can add static parameters by simply specifying their initial values as shown below.

.. code-block:: python

    phase.setStaticParams([0.0,0.0])  # add two static parameters initialized to 0




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
     - Piecewise-Linear (:code:`'FirstOrderSpline'`), Piecewise-Constant (:code:`'BlockConstant'`) 
   * - :code:`'LGL5'`
     - Fifth order Legendre Gauss Lobatto collocation.
       Three states per segment.
     - Simpson's Rule
     - Quadratic-Spline (:code:`'FirstOrderSpline'`), Piecewise-Constant (:code:`'BlockConstant'`)
   * - :code:`'LGL7'`
     - Seventh order Legendre Gauss Lobatto collocation.
       Four states per segment.
     - Unnamed fourth order quadrature method
     - Cubic-Spline (:code:`'FirstOrderSpline'`),
       Natural-Cubic-Spline (:code:`'HighestOrderSpline'`),
       Piecewise-Constant (:code:`'BlockConstant'`)
   * - :code:`'Trapezoidal'`
     - Trapezoidal collocation.
       Two states per segment.
     - Trapezoidal Rule
     - Piecewise-Linear (:code:`'FirstOrderSpline'`), Piecewise-Constant (:code:`'BlockConstant'`)
   * - :code:`'CentralShooting'`
     - Adaptive Dormand Prince 8(7) central shooting method.
       Two states per segment.
     - Trapezoidal Rule
     - Piecewise-Linear (:code:`'FirstOrderSpline'`), Piecewise-Constant (:code:`'BlockConstant'`)

.. _conobj-guide:

Constraints and Objectives
==========================

Before discussing the interface for adding different types of constraints, it is helpful to briefly overview how we represent a phases's variables
when formulating an optimization problem. In general we partition a trajectory with :math:`n` states into each time-varying portion :math:`\vec{V}_i` of the ODE's inputs followed by the
ODE parameters, :math:`\vec{P}`, and the phase's static parameters, :math:`\vec{S}`, below. 

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

The transcription defect constraints, and segment mesh spacing constraints are formulated automatically by the phase object, and users should not
attempt to formulate them on their own. Every other constraint and objective must be specified by the user, in terms of the discrete representation of the trajectory. To simplify this process,
and provide an interface that is invariant to the number of segments, :code:`phase` only allows you to write constraints/objectives that gather inputs from certain "phase regions" in the total variables
vector. A complete list of the currently allowed phase regions is listed below and we will discuss how you can use them in the next section.


.. list-table:: Phase Regions
   :widths: 15 50 35
   :header-rows: 1

   * - Phase Region
     - Description
     - Input Order
   * - :code:`Front`, or :code:`First`
     - Applied to first time-varying-input, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_1,\vec{P},\vec{S}])`
   * - :code:`Back`,or :code:`Last`
     - Applied to last time-varying-input, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_n,\vec{P},\vec{S}])`
   * - :code:`Path`
     - Applied to every time-varying-input, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{P},\vec{S}]),\; i = 1\ldots n`
   * - :code:`InnerPath`
     - Applied to every time-varying-input (excluding the first and last), the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{P},\vec{S}]),\; i = 2\ldots n-1`
   * - :code:`FrontandBack`, or :code:`FirstandLast`
     - Applied to the first and last time-varying-inputs, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_1,\vec{V}_n,\vec{P},\vec{S}])`
   * - :code:`PairWisePath`
     - Applied to every pair of adjacent time-varying-inputs, the ODE parameters and the phase's static parameters.
     - :math:`\vec{f}([\vec{V}_i,\vec{V}_{i+1},\vec{P},\vec{S}]),\; i = 1\ldots n-1`
   * - :code:`ODEParams`
     - Applied only to the ODE parameters.
     - :math:`\vec{f}([\vec{P}])`
   * - :code:`StaticParams`
     - Applied only to the phase's static parameters.
     - :math:`\vec{f}([\vec{S}])`


Equality Constraints
--------------------
Equality constraints of the form :math:`\vec{h}(\vec{x}) = \vec{0}`, can be added to a phase using the :code:`.addEqualCon` method. First we specify 
the phase region to which the constraint will be applied followed by the equality 
constraint itself (an ASSET vector (or scalar) function). Next, we specify which of the indices of time-varying input variables at the phase region,
as well as any ODE parameters and phase's static parameters we wish to forward to the function. In the trivial example below, we are adding a
constraint that enforces that the first time-varying inputs in the trajectory and all of the ODE parameters and static parameters should be equal to zero. 
Custom constraints must be written such that the inputs consist of the time-varying inputs (if any), followed by the ODE parameters (if any), and then the static
parameters (if any). However, the variables inside of a particular variable group (ex::code:`XtUVars`) can be specified in any order so long as it is consistent
with how you have defined your constraint function.

.. code-block:: python

    PhaseRegion = "First"

    def AnEqualCon():
        XtU_OP_SP = Args(13)
        return XtU_OP_SP

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function
    
    phase.addEqualCon(PhaseRegion,AnEqualCon(),XtUVars,OPVars,SPVars)

.. note::

    It should be further emphasized that you do not have to include every variable in a phase region for every constraint.

For example, below we add a constraint involving the second and third state variables from the 
last time-varying state in the trajectory, as well as the first ODE parameter and second static parameter. 



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

Furthermore, when variables from only a single grouping are needed we do not have to pass the others as arguments, as illustrated in the three
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

The previous examples only illustrate the usage of the phase regions that take at most one time-varying input; however, phase regions :code:`"FrontandBack"`, and
:code:`"PairWisePath"` take two time-varying inputs. An example of how to use a two input phase region is shown below. Here we are constraining that the first and last states should be equal
and that the difference between the last and first time of the phase should be equal to a static parameter that we have added to the phase. We only specify which time-varying variables
we want once. The same set is gathered from the first state and last state and forwarded to the function, followed by any ODE parameters (none in this case) 
and static parameters (just the first in this case).

.. code-block:: python

    def FrontBackEqCon():
        X_0,t_0,X_f,t_f,sp0 = Args(15).tolist([(0,6),(6,1),(7,6),(13,1),(14,1)])
    
        eq1 = X_0-X_f
        eq2 = t_f-t_0 - sp0
        return vf.stack(eq1,eq2)
     

    XtUVars = range(0,7)  # indices of all states and time
    SPVars  = [0]  # first static parameter
    # Constrain first and last states to be equal and
    # constrain Delta Time over the phase (tf-t0) to be equal to the first static parameter
    phase.addEqualCon("FirstandLast",FrontBackEqCon(),XtUVars,[],SPVars)

In addition to the general methods of adding equality constraints illustrated in the previous examples, there are several additional methods to simplify the 
definition of commonly occurring types of constraints. By far the most commonly used is the :code:`.addBoundaryValue` method, which simply adds a constraint that the specified
variables should be equal to some vector of constants (can be a 1-D numpy array or a python list). This method is typically used to 
enforce known initial and terminal conditions on a phase.

.. code-block:: python
    
    XtUVars = [1,3,9]
    Values  = [np.pi,np.e,42.0]
    phase.addBoundaryValue("First",XtUVars,Values)

    OPVars  = [0]
    Values  = [10.034]
    phase.addBoundaryValue("ODEParams",OPVars,Values)

    SPVars = [0,1]
    Values  = np.array([1.0,4.0])
    phase.addBoundaryValue("StaticParams",SPVars,Values)


Additionally, you can also use the :code:`addDeltaVarEqualCon` method
to constrain changes in variables from the :code:`"First"` to :code:`"Last"` phase regions to a specified value. This could, for example,
be used to enforce a fixed duration for the phase by supplying the index for time (:code:`"6"` in this case). However, constraining the delta time is
so common that we also provide the :code:`addDeltaTimeEqualCon` method to do just that.

.. code-block:: python
    
    # Constrain change in 0th state variable from first to last state to be = 1.0
    phase.addDeltaVarEqualCon(0,1.0)
    # This does the same as the following

    DeltaEqualCon= Args(2)[1]-Args(2)[0] -1.0
    phase.addEqualCon("FirstandLast",DeltaEqualCon,[0])


    ## These do the same thing, constraining the elapsed time over the phase to be = 3.0
    phase.addDeltaVarEqualCon(6,3.0)
    phase.addDeltaTimeEqualCon(3.0) #Time is special and has its own named method

    # Both are equivalent to the following
    DeltaEqualCon= Args(2)[1]-Args(2)[0] -3.0
    phase.addEqualCon("FirstandLast",DeltaEqualCon,[6])





Inequality Constraints
----------------------
Adding general inequality constraints, using :code:`.addInequalCon`, works exactly the same as it did for :code:`.addEqualCon`. The only difference
is that our functions should be constraints of the form :math:`\vec{g}(\vec{x}) \leq \vec{0}`. In other words, we assume that our function is in the feasible region whenever
its value is negative. For example, if we wanted to add a constraint specifying that all of the initial time-varying input variables, ODE parameters, and the phase's static
parameters should be positive, we could implement that as shown below.

.. code-block:: python

    PhaseRegion = "First"

    def AnInequalCon():
        XtU_OP_SP = Args(13)
        return -1.0*XtU_OP_SP

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function
    
    phase.addInequalCon(PhaseRegion,AnInequalCon(),XtUVars,OPVars,SPVars)

    # Other signatures follow the same rules as covered for addEqualCon
    phase.addInequalCon("Path", Args(4).sum(),[0,1,2],[],[1])
    phase.addInequalCon("Back",  Args(3).squared_norm()-1,[3,4,5])
    phase.addInequalCon("StaticParams",1-Args(2).norm(),[0,1])


However, it can be somewhat cumbersome to write many of the types of inequality constraints that you will encounter using this generalized method, thus
we offer many simplified alternatives which we now discuss.

The simplest type of inequality constraint we can apply are bounds on the variables. These can be added using the :code:`.addLower/Upper/LUVarBounds` methods as 
shown below, these can be applied to any of the single time-varying input phase regions or the parameters. For any method, we can also specify a positive scale factor that will be 
applied to the final bounding function. This can help scale an ill-conditioned bound but will not change the meaning of the constraint.


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

In addition to placing bounds on variables, you can also place bounds on the outputs of ScalarFunctions of the variables. This
is accomplished using the :code:`.addLower/Upper/LUFuncBound` methods as shown below. In this example we are showing various ways to bound the norm
of all of the controls (variables :code:`[7,8,9]` for this contrived ODE) to be between 0 and 1.0.

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


These methods can be applied to any ScalarFunction you wish to bound; however, the examples above
that bound the :code:`norm` or :code:`squared_norm` are so common that we also provide methods that do just that. Below, we use the :code:`.addLower/Upper/NormBound`
and :code:`.addLower/Upper/SquaredNormBound` methods that accomplish the same tasks as the previous code block.

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


Similar to how we can place equality constraints on the change in a variable from the beginning to end of
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
The simplest type of objective function that we can add to a phase is a state objective. It is a ScalarFunction that we
wish to directly minimize that takes some or all of the variables at a phase region. Note, if you are trying to maximize something you should
multiply it's value by a negative constant, as ASSET interprets all objective values as values to be minimized. Generalized state objectives can be added to a phase using the :code:`.addStateObjective` function as shown below. The same rules governing
:code:`.addEqualCon`, and :code:`.addInequalCon` apply here to all possible permutations to the inputs of :code:`.addStateObjective`. The only exception being that the function
must be an ASSET ScalarFunction.

.. code-block:: python

    def AStateObjective():
    XtU_OP_SP = Args(13)
    return XtU_OP_SP.norm()  ## An Asset Scalar Function

    PhaseRegion = "Back"
    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function

    phase.addStateObjective(PhaseRegion,AStateObjective(),XtUVars,OPVars,SPVars)

In addition to the general methods, we also provide two more specialized methods that encompass two of the most common types of state objectives.

The first is the :code:`.addValueObjective` method which simply adds an objective function specifying that we want to minimize the value of one the variables
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

The second is the :code:`.addDeltaVarObjective` which adds an objective to minimize the change in the value of some variable across the phase multiplied by 
a scale factor. As before, to maximize the change, make the scale factor negative.

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
The other common type of objective functions that we can add to a :code:`phase` are integral objectives of the form.

.. math::
   
   \int_{t_0}^{t_f} f([\vec{X}(t),t,\vec{U}(t),\vec{P},\vec{S}]) dt


To add an integral objective, we provide a scalar integrand function to the :code:`phase` using the :code:`.addIntegralObjective` method. 
The quadrature method used to approximate the integral will be depend on the current transcription type and are given in table 1.
When adding integral objectives as shown below, we only need to provide the integrand function and the
indices from the various variable groupings we want to forward to the integrand (ie: no phase region is needed). 

.. code-block:: python

    def AnIntegrand():
        XtU_OP_SP = Args(13)
        return XtU_OP_SP.norm()  ## An Asset Scalar Function

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,2)   # indices of the phase Static Parameters (indexed from 0) we want to forward to our function


    # Signature if variables of all types are needed by integrand
    phase.addIntegralObjective(AnIntegrand(),XtUVars,OPVars,SPVars)

    # Signature if only state,time, and control variables needed by integrand
    phase.addIntegralObjective(Args(3).norm(),[7,8,9])

    # All integrands are minimized, so to maximize, multiply by negative number
    phase.addIntegralObjective(-10.0*Args(4).norm(),[1,2,3,7])
    


Integral Parameter Functions
----------------------------

The final class of functions that we can add to a phase are what we call "integral parameter functions". These are used along with static parameters
to facilitate integral constraints on a :code:`phase`. An integral parameter function is a special equality constraint of the form. 

.. math::
   
   \int_{t_0}^{t_f} f([\vec{X}(t),t,\vec{U}(t),\vec{P},\vec{S}_{\not k}]) dt - s_k = 0

Essentially, this constraint will force the value of one of the static parameters to be equal to the integral of a user specified function. 
One can then place constraints on this static parameter using any of the previously discussed methods.
Adding in an integral parameter function, using :code:`.addIntegralParamFunction` works essentially the same as :code:`.addIntegralObjective`, except we also provide the index
of the static parameter the integral value will be assigned to as the last argument.

.. code-block:: python

    def AnIntegrand():
        XtU_OP_SP = Args(12)
        return XtU_OP_SP.norm()  ## An Asset Scalar Function

    XtUVars = range(0,10)  # indices of state, time, and control variables at the PhaseRegion we want to forward to our function
    OPVars  = range(0,1)   # indices of the ODE Parameters (indexed from 0) we want to forward to our function
    SPVars  = range(0,1)   # indices of the phase Static Parameters (indexed from 0), NOT INCLUDING THE ONE WE ARE ASSIGNING THE INTEGRAL TOO
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

After constructing a phase, supplying an initial guess, and adding constraints/objectives, we can now use PSIOPT to solve or optimize
the trajectory. The settings of the optimizer can be manipulated through a reference to PSIOPT attached to the phase object. However, calls to
the optimizer are handled through the phase itself as shown below. Both of these topics are handled in more details in the section on :ref:`PSIOPT <psiopt-guide>`.


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


After finding a solution, we can retrieve the converged trajectory using the :code:`.returnTraj` method of the :code:`phase`. 
Note the trajectory is returned as a python list where each element is a full-ode input (ie: :math:`[\vec{X}_i,t_i,\vec{U}_i,\vec{P}]`) at each point in time along the trajectory.
You may also return the trajectory in the form of an :code:`oc.LGLInterpTable` so that it can be sampled as a smooth function of time. See the section on :ref:`LGLInterpTable and InterpFunction` for more details.
If you added static parameters to the :code:`phase`, these can be retrieved using :code:`.returnStaticParams`. Finally, you can also retrieve an estimate for the co-states
of an optimal control problem AFTER it has been optimized. These could then be used as the initial guess to an indirect form of the same optimization problem.

.. code-block:: python
    
    Traj = phase.returnTraj()

    ## Output trajectory has same format as input
    for XtUP in Traj:
        XtUP[0:6]  # The state variables,X
        XtUP[6]    # The time,t
        XtUP[7:10] # The control variables,U
        XtUP[10]   # The ODE parameter,P

    Tab = phase.returnTrajTable()  ## As an LGL interp table

    StatParams = phase.returnStaticParams()

    CostateTraj = phase.returnCostateTraj() #

    for Ct in CostateTraj:
        C[0:6] # The Costates associated with X
        C[6]   # The time
        


Additionally, should you want to refine the mesh spacing of the trajectory after a solution, it is not necessary to create an entirely new :code:`phase`.
Instead, you can use the :code:`.refineTraj` methods as shown below. The simplest form of refinement can be accomplished using the :code:`.refineTrajManual` methods. In general these work exactly
the same as the :code:`.setTraj` methods except they use the currently loaded trajectory to interpolate the new mesh. 
The second option is the :code:`.refineTrajEqual` method, which will attempt to refine the trajectory such the estimated error across all segments is equal. 
Fortunately, ASSET's run-time scales basically linear in the number of segments, so it is often a viable strategy to just double or quadruple 
(or more) the number of segments, re-optimize and call it a day. Beginning in version 0.1.0, we also now have a closed loop adaptive mesh refinement method that will automatically update
the spacing and number of segments to meet desired error tolerances. See the  :ref:`Adaptive Mesh Refinement Tutorial <mesh-guide>` tutorial for more details.

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




Miscellaneous Topics
====================


Shooting Method
---------------
When using the Central Shooting transcription, under the hood, a phase uses an integrator for the corresponding ODE to formulate the shooting constraints and its derivatives.
This :code:`integrator` is always configured to use the :code:`"DOPRI87"` integration scheme, but users can modify the tolerances as well as the minimum and
maximum step sizes of the integrator to improve performance or increase accuracy. Users can access this :code:`integrator` using the :code:`.integrator` field of the phase and then
modify its settings just as was shown in the :ref:`integrator tutorial <integrator-guide>`. Note that we set the default step size of the integrator attached to a phase to 0.1. For fastest performance
you should modify this to be something near what you anticipate the real average step size to be when integrating your ODE.

.. code-block:: python

    phase = ode.phase("CentralShooting")

    phase.integrator.setAbsTol(1.0e-13)  ## Modify tolerances of adaptive step size algorithm

    DefStep = .05
    MinStep = .00001
    MaxStep = 2.0

    phase.integrator.setStepSizes(DefStep,MinStep,MaxStep)  # Modify default,minimum and maximum step sizes.


Control Rate Constraints
-------------------------
You may have noticed from the previous examples, that we do not provide an explicit method for constraining control rates :math:`\dot{\vec{U}}`. 
However, this can be accomplished manually by using a custom constraint with phase region :code:`'PairWisePath'`.
For example, if we wanted to bound the rates of the control variables (:code:`[7,8,9]` in this example) to be between -1 and 1, we could do so with the following code.
:code:`'PairWisePath'` will, as the name suggests, call our :code:`URateBound` function at every sequential pair of time-varying states in the trajectory, thus allowing us to bound a linear estimate of
the control rates from the times and values of the controls.

.. code-block:: python
    
    def URateBound(LBoundVec,UBoundVec):
        tUtU = Args(8)
        ti,Ui = tUtU.head(4).tolist([(0,1),(1,3)])
        tip1,Uip1 = tUtU.tail(4).tolist([(0,1),(1,3)])
    
        h = tip1-ti  
        Urate = (Uip1-Ui)/(h)
    
        UpperBound = Urate - UBoundVec
        LowerBound = LBoundVec - Urate
    
        return vf.stack(UpperBound,LowerBound)

    phase.addInequalCon("PairWisePath",URateBound(-np.ones(3),np.ones(3)),[6,7,8,9])
    
It should be noted that for the higher order collocation methods (:code:`LGL5` and :code:`LGL7`), the controls are piece-wise quadratic or cubic, so this will not be an exact constraint.
However, in most cases bounding the local linear rate in this way will work just fine. However, we should also note that you should not apply control rate
constraints like the above to phases with :code:`'BlockConstant'` control parameterization. It will "work" (as in not throw errors or die violently), but it will very likely result in a problem
that is over constrained and structurally singular. Finally, if you need more accurate estimates of the control rates, you always have to option of writing a new ODE where the controls in question are promoted
to state variables and their rates become the new controls.

    
What happens if I over constrain my problem?
--------------------------------------------
We do check that the total number of equality constraints added to a problem (dynamics, mesh-spacing, and user equality constraints) is less than 
the total number of variables. However, most cases of over constraining are much more subtle than this and happen when you have added redundant or conflicting 
constraints. We do not explicitly check for this at the moment. An example of what we mean by this is illustrated below. Let's say we needed to constrain the duration
of a phase to be some fixed value, and for the initial time to be equal to 0. We could add a boundary value to the first state in the trajectory to fix the initial time.
Since the initial time is constrained to be 0, fixing the final time to be :code:`dt` will constrain the phase's duration as we expect. However, if we were to then accidentally use :code:`.addDeltaTimeEqualCon`
to fix the phase duration as well, then the problem is over constrained. Thus, you should either fix the initial and final times, or fix the initial time and duration, but not both.

.. code-block:: python

    # time is variable 6 for this problem

    dt = 1.0

    phase.addBoundaryValue("First",[6],[0.0])
    phase.addBoundaryValue("Back", [6],[dt])

    #. Other things that make you forget what you have already done
    #.
    #.
    phase.addDeltaTimeEqualCon(dt)  # BAM!! Over constrained

    phase.solve()  # Flying red numbers from the output scroll

For problems with controls, mistakes like this will typically not result in an excess of equality constraints, and thus your only indication that something is wrong will be
poor or erratic performance by the optimizer. Sometimes the optimizer's pivoting perturbation will be able to cope with redundant constraints and return solutions, other times it will
diverge immediately. In conclusion, don't over constrain your problems...

What happens if I add multiple objectives?
------------------------------------------

A phase has no restriction of the number of objectives that may be added. If multiple objectives are added, the optimizer will implicitly sum all of their values.


Bad Initial Guesses 
-------------------

By default we take whatever the supplied initial guess to a phase is and use the specified transcription scheme's interpolation method to generate the initial mesh.
For compressed collocation methods, when the initial guess is very poor, this can induce osculations in the initial interpolated mesh that are not present in the initial guess
. This can be avoided by instead interpolating the initial mesh linearly from the supplied initial guess. You can specify the method for the initial interpolation of a phase as shown below.
If the user supplied initial guess is linear or constant, we recommend using the linear interpolation method.

.. code-block:: python
       

       LerpIG = False
       #Interpolate initial mesh using transcriptions interpolation method (same as default)
       phase = ode.phase("LGL3",TrajIG,nsegs,LerpIG) 
       phase = ode.phase("LGL3",TrajIG,nsegs) # Above is the same as this

       LerpIG = True
       #Interpolate initial mesh using linear interpolation
       phase = ode.phase("LGL3",TrajIG,nsegs,LerpIG)
       

       

Reintegrating Solutions
-----------------------
Reintegration of a phase's trajectory can be accomplished using an ODE's integrator and the tabular
form of the solution returned by the :code:`phase.returnTrajTable` method. To do this, supply the :code:`LGLInterpTable`
object returned by :code:`returnTrajTable` to the constructor of the ODE's integrator type. This will automatically
initialize the integrator to use the control history stored in the trajectory data as a time dependent control law
when integrating. If the control history of the phase is not :code:`"BlockConstant"`, you can then call :code:`integrate_dense` to
integrate from the initial full ODE input in the returned trajectory to the final time in the trajectory. This will
still work if the control history was :code:`"BlockConstant"`, but the result may have small local errors caused by the instantaneous jumps in the control
history at the states where segments adjoin. This can be eliminated by using the second method, which integrates
precisely between each time in the converged trajectory.

.. code-block:: python

    ConvTraj = phase.returnTraj()
    Tab  = phase.returnTrajTable()
    
    integ = ode.integrator(.1,Tab)  # provide the returned table as arg to integrator
    integ.setAbsTol(1.0e-13)
    
    # recall time is variable 6 for this ODE

    ## Do this for non-BlockConstant control or if you don't care about exact accuracy
    ## Integrate from initial ODE input to final time
    ReintTraj1 = integ.integrate_dense(ConvTraj[0],ConvTraj[-1][6])
    
    ## This is to be preferred if control is BlockConstant
    ## Integrate precisely between each time so integrator doesnt see instantaneous jump in control
    ReintTraj2 = [ConvTraj[0]]    
    for i in range(0,len(ConvTraj)-1):
        Next = integ.integrate_dense(ReintTraj2[-1],ConvTraj[i+1][6])[1::]
        ReintTraj2+=Next


.. _phaseremove-guide:

Referencing and Removing Constraints
------------------------------------

When adding any of the 5 types of constraints/objectives covered :ref:`previously <conobj-guide>`, an integer identifier or list of integers is returned by the 
method. This identifier can be used to remove a constraint/objective from the problem. This can be quite useful when you
want to express some homotopic or continuation scheme without having to create a new phase at each step. Given the identifier for
a function of a certain type, it can be removed from the phase using the corresponding :code:`.remove#####(id)` method as shown below.


.. code-block:: python

    ###########################
    ## Ex. Equality Constraint
    edx1 = phase.addBoundaryValue("Front",range(0,5),np.zeros((5)))
    edx2 = phase.addEqualCon("Path",Args(3).norm()-1.0,[8,9,10]) 
    edx3 = phase.addDeltaVarEqualCon(6,1.0)

    ## Removal order doesn't matter
    phase.removeEqualCon(edx1)
    phase.removeEqualCon(edx3)
    phase.removeEqualCon(edx2)


    #############################
    ## Ex. Inequality Constraint
    idx1 = phase.addInequalCon("Path", Args(4).sum(),[0,1,2],[],[1])
    idx2 = phase.addLUVarBound("StaticParams",0,-1.0,1.0)
    idx3 = phase.addLUFuncBound("Path",Args(3).norm(),[8,9,10],0,1)

    phase.removeInequalCon(idx2)
    phase.removeInequalCon(idx3)
    phase.removeInequalCon(idx1)

    #######################
    ## Ex. State Objective
    sdx1 = phase.addStateObjective("Back",Args(3).squared_norm(),[0,1,2])
    sdx2 = phase.addValueObjective("Back",6,1.0)
    sdx3 = phase.addDeltaTimeObjective(1.0)

    phase.removeStateObjective(sdx3)
    phase.removeStateObjective(sdx2)
    phase.removeStateObjective(sdx1)


    ##########################
    ## Ex. Integral Objective
    intdx1 = phase.addIntegralObjective(Args(3).norm(),[7,8,9])

    phase.removeIntegralObjective(intdx1)

    ##########################
    ## Ex. Integral Parameter Function
    ipdx1 = phase.addIntegralParamFunction(Args(3).norm(),[7,8,9],IntSPVar)

    phase.removeIntegralParamFunction(ipdx1)




Retrieving Constraint Violations and Multipliers
------------------------------------------------
Immediately after a call to PSIOPT, users can retrieve the constraint violations and Lagrange multipliers associated with user applied constraints
as well as the dynamics transcription and control spline constraints. This can be helpful when debugging non-converging problems. For equality and inequality constraints, constraint
violations and multipliers are retrieved by supplying a constraint functions id to the phase's :code:`.return####Vals(id)` and :code:`.return####Lmults(id)` methods as shown below.
In all cases, the violations/multipliers are returned as a list of numpy arrays, each of which contains the output/multipliers associated with each call to the function inside of
the optimization problem. For constraints applied only at a single state, the returned list will contain only one numpy array. In general, for path constraints, the list will contain the same number of elements
as the returned trajectory, and the constraint violations in the ith element will be associated with calling the constraint with variables from the ith state as inputs. Note that for inequality constraints, the return values do not have slacks applied, thus
negative values indicate that the constraint is in the feasible region and positive values indicate that the constraint is in the infeasible region.

.. code-block:: python

    edx = phase.addBoundaryValue("Front",range(0,5),np.zeros((5)))
    idx = phase.addLUVarBound("Path",0,-1.0,1.0)

    phase.optimize()

    ecvals = phase.returnEqualConVals(edx)
    ecmults = phase.returnEqualConLmults(edx)

    print(ecvals[0])
    print(ecmults[0])


    icvals  = phase.returnInequalConVals(idx)
    icmults = phase.returnInequalConLmults(idx)

    for icval,icmult in zip(icvals,icmults):
        print(icval)
        print(icmult)


Transcription defect constraint violations can be retrieved with the :code:`.returnTrajError()` method. Here each element in the returned list 
is a numpy array containing the subset of the defect equality constraint errors roughly attributable to the given time. This association is not rigorous or exact,
and is only meant as a guide for roughly determining where the defect constraints become difficult to satisfy. Furthermore, this error is only related to solution
of the optimal control problem, and is not the mesh error estimated by the adaptive mesh refinement scheme (though the two can be correlated). In our formulation, co-states are interpolated directly
from the Lagrange multipliers associated with the transcription defect constraints, so users can use the previously discussed :code:`.returnCostateTraj()` to examine the behavior of the multipliers.


.. code-block:: python

    ETraj = phase.returnTrajError()

    for Et in ETraj:
        Et[0:6]  # The defect error
        Et[6]    # roughly at time t


Finally, for the you can also retrieve the constraint values and multipliers for the control spline regularization functions as shown below. If no spline constraints are applied,
the returned lists will be empty.

.. code-block:: python
    
    Usm = phase.returnUSplineConLmults()
    Usc = phase.returnUSplineConVals()






    

