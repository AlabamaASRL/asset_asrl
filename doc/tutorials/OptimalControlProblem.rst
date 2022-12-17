================================
Optimal Control Problem Tutorial
================================

The previous section on phases covers everything you need to know about formulating and solving
single phase optimal control problems. In this section, we will cover how you can link these 
phases together into a multi phase optimal control problem using the :code:`oc.OptimalControlProblem` type.

As we walk through OptimalControlPoblems API, we will make the trivial ODEs shown below.

.. code-block:: python

    import numpy as np
    import asset_asrl as ast

    vf        = ast.VectorFunctions
    oc        = ast.OptimalControl
    Args      = vf.Arguments

    class DummyODE(oc.ODEBase):
        def __init__(self,xv,uv,pv):
            args = oc.ODEArguments(xv,uv,pv)
            super().__init__(args.XVec(),xv,uv,pv)
        
        
    # 6 states indexed [0,1,2,3,4,5], time index 6, no controls or ODE params
    odeX   = DummyODE(6, 0, 0)

    # 6 states indexed [0,1,2,3,4,5], time index 6, 3 controls [7,8,9], no ODE params
    odeXU  = DummyODE(6, 3, 0)

    # 7 states indexed [0,1,2,3,4,5,6], time index 7, 3 controls indexed [8,9,10], one ODE param
    odeXUP = DummyODE(7, 3, 1)



Initialization
==============

To start, we will create 6 phases (2 of each ODE) type, as shown below that we will then link together inside of
a single optimal control problem. Though not shown here, each phase may be constructed using methods we discussed in the
previous section. 

.. code-block:: python

    phase0 = odeX.phase("LGL3")
    phase1 = odeX.phase("LGL3")

    phase2 = odeXU.phase("LGL3")
    phase3 = odeXU.phase("LGL3")

    phase4 = odeXUP.phase("LGL3")
    phase5 = odeXUP.phase("LGL3")

    phase4.setStaticParams([0])
    phase5.setStaticParams([0])

    #. setup phases 0 through 5
    #.
    #.
    #.
    #.

We can then create an OptimalControlProblem, and add our phases using the addPhase method.
The individual phases in an ocp, MUST Be unique objects. The software will detect
if you attempt to add the same phase to an ocp twice and throw an error. The commented
out line below will throw an error because the specific phase5 object has already been added to
the ocp.

.. code-block:: python

    ocp  = oc.OptimalControlProblem()

    ocp.addPhase(phase0)
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    ocp.addPhase(phase5)

    #ocp.addPhase(phase5)  #phases must be unique, adding same phase twice will throw error

You can access the phases in an ocp using the ocp.Phase(i) method where i
is the index of the phase in the order they were added. If the phase is created
elswhere in the script you can manipulate it throught that object or via
the .Phase(i) method as shown below. Note, phases are large stateful objects and we
do not make copies of them , thus ocp.Phase(0) and phase0 are the EXACT
same object. Be careful not to apply duplicate constraints to the same phase accidentally
as WE DO NOT CHECK FOR THIS.

.. code-block:: python

    ocp.Phase(0).addBoundaryValue("Front",range(0,6),np.zeros((6)))

    # Equivalent to above,make sure you dont accidentally do both.
    # phase0.addBoundaryValue("Front",range(0,6),np.zeros((6)))

Additionally, you make access the list of phases already added to an ocp using the .Phases field
of the object. This can allow you to iterate over all phases to apply similar constraints/objectives
to some or all of the phases as shown below.

.. code-block:: python

    for phase in ocp.Phases:
        phase.addDeltaTimeObjective(1.0)


As a general rule of thumb, any constraint or objective that can be applied to the individual phases to represent your goal, should be
 , and not using the OptimalControlPorblem api that we are about to cover in section. For example, if our intent was to minimize
the total time elpased time of all of our phases, applying a addDeltaTimeObjective to every phase should be preferred to an equivalent formuation using 
LinkObjectives.

Analogous to the concept of a phase's static parameters, you may also add additional free variables that we call Link Parameters to an ocp as shown below.

.. code-block:: python
    ocp.setLinkParams(np.ones((15)))



Link Constraints and Objectives
===============================
Application of link objectives and constraints in an optimal control problem, is built upon the concept of phase regions
and indexing we covered in phase.

Link Equality Constraints
-------------------------
An optimal control problems link equality constraints of the form f(x) =0, that take as arguments, the variables from one or phases
at specified phase regions, as well as Link parameters (if any). A link equality constraint can be added to the phase using one the 
addLinkEqualCon method. The most general way to link two phases with an inequality constraint is shown below. This contrived example is
enforcing continuity the last time-varying state variables and in phase0 and the first-time varying state variables and parameters in phase1.
For illustrative pourposes we also multply the result by the 0th link parameter. Our constraint function should be formulated to expect
all arguments specified for phase0 (V0), followed by all specified for phase1 (V0), followed by the link parameters (Lvar).

.. code-block:: python

    def ALinkEqualCon():
        V0,V1,Lvar = Args(27).tolist([(0,13),(13,13),(26,1)])
        return (V0-V1)*Lvar


    XtUvars0 = range(0,11)
    OPvars0 = [0]
    SPvars0 = [0]

    XtUvars1 = range(0,11)
    OPvars1 = [0]
    SPvars1 = [0]

    LPvars  = [0]

    ## Use index in the of the phase in the ocp to specify each phase
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        0,'Last', XtUvars0,OPvars0,SPvars0,
                        1,'First',XtUvars1,OPvars1,SPvars1,
                        LPvars)

    ## Same as above, but use the phase objects themselves to specify each phase
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        phase0,'Last', XtUvars0,OPvars0,SPvars0,
                        phase1,'First',XtUvars1,OPvars1,SPvars1,
                        LPvars)
    
    ## Same as above
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        ocp.Phase(0),'Last', XtUvars0,OPvars0,SPvars0,
                        ocp.Phase(1),'First',XtUvars1,OPvars1,SPvars1,
                        LPvars)

If the constraint function does not need any Link parameters, they may be ommitted from the function call.
Furthermore,




Link Inequality Constraints
---------------------------


Link Objectives
---------------




.. math::

   \vec{x} = \begin{bmatrix}
              \vec{x}^1
              \vdots
              \vec{x}^m
              \vec{L}
              \end{bmatrix}

   \vec{x}^j = \begin{bmatrix}
              \vec{V}_1^j     \\
              \vec{V}_2^j     \\
              \vdots        \\
              \vec{V}_{n-1}^j \\
              \vec{V_n}^j     \\
              \vec{P}^j       \\
              \vec{S}^j       \\
             \end{bmatrix}
    \quad \quad \text{where} \quad \vec{V}_i^j = [\vec{X}_i^j,t_i^j,\vec{U}_i^j]


