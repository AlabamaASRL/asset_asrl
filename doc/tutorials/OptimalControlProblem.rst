================================
Optimal Control Problem Tutorial
================================

The previous section on :ref:`phases <phase-guide>` covers everything you need to know about formulating and solving
single phase optimal control problems. In this section, we will cover how you can link these 
phases together into a multi phase optimal control problem using the :code:`oc.OptimalControlProblem` type.

As we walk through the :code:`OptimalControlPoblem` API, we will make use of the trivial ODEs shown below.

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
previous :ref:`section <phase-guide>`. 

.. code-block:: python

    phase0 = odeXUP.phase("LGL3")
    phase1 = odeXUP.phase("LGL3")

    phase0.setStaticParams([0.0])
    phase1.setStaticParams([0.0])

    phase2 = odeXU.phase("LGL3")
    phase3 = odeXU.phase("LGL3")

    phase4 = odeX.phase("LGL3")
    phase5 = odeX.phase("LGL3")

    phase4.setStaticParams([0])
    phase5.setStaticParams([0])

    #. setup phases 0 through 5
    #.
    #.
    #.
    #.

We can then create an :code:`OptimalControlProblem` (here named :code:`ocp`), and add our phases using the :code:`addPhase` method.
The individual phases in an :code:`OptimalControlProblem` MUST Be unique objects. The software will detect
if you attempt to add the same phase to an :code:`OptimalControlProblem` twice and throw an error. The commented
out line below will throw an error because the specific :code:`phase5` object has already been added to
the :code:`OptimalControlProblem`.

.. code-block:: python

    ocp  = oc.OptimalControlProblem()

    ocp.addPhase(phase0)
    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)
    ocp.addPhase(phase4)
    ocp.addPhase(phase5)

    #ocp.addPhase(phase5)  #phases must be unique, adding same phase twice will throw error

You can access the phases in an :code:`OptimalControlProblem` using the :code:`ocp.Phase(i)` method where :code:`i`
is the index of the phase in the order they were added. If the phase is created
elsewhere in the script you can manipulate it through that object or via
the :code:`.Phase(i)` method as shown below. Note, phases are large stateful objects and we
do not make copies of them , thus :code:`ocp.Phase(0)` and :code:`phase0` are the EXACT
same object. Be careful not to apply duplicate constraints to the same phase.

.. code-block:: python

    ocp.Phase(0).addBoundaryValue("Front",range(0,6),np.zeros((6)))

    # Equivalent to above,make sure you dont accidentally do both.
    # phase0.addBoundaryValue("Front",range(0,6),np.zeros((6)))

Additionally, you may access the list of phases already added to an :code:`ocp` using the :code:`.Phases` field
of the object. This can allow you to iterate over all phases to apply similar constraints/objectives
to some or all of the phases as shown below.

.. code-block:: python

    for phase in ocp.Phases:
        phase.addDeltaTimeObjective(1.0)


As a general rule of thumb, any constraint or objective that can be applied to the individual phases to represent your goal, should be, 
and not with the :code:`OptimalControlProblem` API that we are about to cover in the next section. For example, if our intent was to minimize
the total time elapsed time of all of our phases, applying :code:`addDeltaTimeObjective` to every phase should be preferred to an equivalent formulation using 
Link Objectives.

Analogous to the concept of a phase's static parameters, you may also add additional free variables that we call "link parameters" to an :code:`ocp` as shown below.

.. code-block:: python

    ocp.setLinkParams(np.ones((15)))


.. _link-guide:

Link Constraints and Objectives
===============================
Application of link objectives and constraints in an :code:`OptimalControlProblem`, is built upon the concept of phase regions
and indexing we covered in phase :ref:`tutorial <phase-guide>`. The total variables vector, :math:`\vec{x}`, consists of those defined for each phase, :math:`\vec{x}^{j}`, followed by
the link parameters, :math:`\vec{L}`. 

.. math::

   \vec{x} = \begin{bmatrix}
              \vec{x}^1\\
              \vdots\\
              \vec{x}^m\\
              \vec{L}\\
              \end{bmatrix}
       \quad \quad \text{where} \quad
   \vec{x}^j = \begin{bmatrix}
              \vec{V}_1^j     \\
              \vec{V}_2^j     \\
              \vdots        \\
              \vec{V}_{n-1}^j \\
              \vec{V_n}^j     \\
              \vec{P}^j       \\
              \vec{S}^j       \\
             \end{bmatrix}
    \quad \quad \text{and} \quad \vec{V}_i^j = [\vec{X}_i^j,t_i^j,\vec{U}_i^j]

Linking constraints and objectives are then functions of the form shown below. They may take as arguments the first
and/or last time-varying-states as well as any parameters from any number of the constituent phases in any specified order, followed by any
extra link parameters.

.. math::

    \vec{f}([\vec{V}_{1\lor n}^k,\vec{P}^k,\vec{S}^k,\ldots \vec{L}]) \quad  k \in [1,\ldots m]


Link Equality Constraints
-------------------------
A link equality constraint of the form :math:`\vec{h}(\vec{x}) = \vec{0}` can be added to the phase using the 
:code:`.addLinkEqualCon` method. The most general way to link two phases with an equality constraint is shown below. This contrived example is
enforcing continuity between the last time-varying state variables and in :code:`phase0` and the first-time varying state variables and parameters in :code:`phase1`.
To illustrate the expected order of arguments we also multiply the result by the 0th link parameter. Our constraint function should be formulated to expect
all arguments specified for :code:`phase0` ( :code:`V0`), followed by all specified for :code:`phase1` ( :code:`V0`), followed by the link parameter ( :code:`Lvar`).

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
    ## You can used phases or integers with any signature, but do not mix them in a single call
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        phase0,'Last', XtUvars0,OPvars0,SPvars0,
                        phase1,'First',XtUvars1,OPvars1,SPvars1,
                        LPvars)
    
    ## Same as above
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        ocp.Phase(0),'Last', XtUvars0,OPvars0,SPvars0,
                        ocp.Phase(1),'First',XtUvars1,OPvars1,SPvars1,
                        LPvars)

If the constraint function does not need any link parameters, they may be omitted from the function call. Additionally,
as was the case for the methods in phase, it is only necessary to provide the specific variables needed from each variable group
required to formulate your custom constraints. The ordering of indices within each group can also be arbitrary so long is it is mathematically
consistent with the constraint you have defined.

.. code-block:: python

    def ALinkEqualCon():
        VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
        return VS0.dot(VP1)


    XtUvars0 = [3,4,5]
    SPvars0  = [0]

    XtUvars1 = [3,1,2]
    OPvars1 = [0]

    ## Enforce that the dot product of the specified variables from each phase region =0
    ocp.addLinkEqualCon(ALinkEqualCon(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])

Furthermore, if your function only requires variables from a single group in each phase, you may omit the others from the function call.

.. code-block:: python

    SomeFunc = Args(6).head(3).cross(Args(6).tail(3))
    ## Only need XtUVars from phases 2 and 3 at the specified regions
    ocp.addLinkEqualCon(SomeFunc,
                        2,'Last', range(0,3),
                        3,'First',range(0,3))


    SomeOtherFunc = Args(2).sum()-1
    # Only needs ODEparams from phases 0 and 1
    ocp.addLinkEqualCon(SomeOtherFunc,
                        0,'ODEParams', [0],
                        1,'ODEParams', [0])

If you need to express a constraint in terms of more than 2 phases at a time, you can utilize the method shown below. Here we pass a list
of tuples each containing all of the arguments needed to specify the variables from a phase region.

.. code-block:: python
    
    def TriplePhaseLink():
        X0,X1,X2 = Args(9).tolist([(0,3),(3,3),(6,3)])
    
        return vf.sum(X0,X1,X2)


    XtUvars = range(3,6)
    SPvars = []  # none needed, leave empty but still pass it in
    OPvars = []  # none needed, leave empty but still pass it in
    LPvars = []  # none needed, leave empty but still pass it in

    # List of tuples of the variables and regions needed from each phase
    ocp.addLinkEqualCon(TriplePhaseLink(),
                        [(3,'First', XtUvars,OPvars,OPvars),
                         (4,'First', XtUvars,OPvars,OPvars),
                         (5,'First', XtUvars,OPvars,OPvars)],
                        LPvars)

Finally, if you need to enforce an equality constraint that only involves the link parameters, you can use the :code:`.addLinkParamEqualCon`
function as shown below.

.. code-block:: python

    # Enforce that the norm of first 3 link params is 1
    LPvec = [0,1,2]
    ocp.addLinkParamEqualCon(Args(3).norm()-1.0,LPvec)

    # Apply same constraint to multiple groups of 3 link params
    LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
    ocp.addLinkParamEqualCon(Args(3).norm()-1.0,LPvecs)




The previously discussed methods can be used define rather complicated phase linkages.
However, in most cases we just want to enforce simple continuity constraints
between certain variables in each phase. This can be accomplished using the :code:`addDirectLinkEqualCon` function as shown below.

.. code-block:: python

    # Enforce that variables XtUvars [3,4,5] in the last state of phase0
    # are equal to the same variables in the first state of phase1
    ocp.addDirectLinkEqualCon(0,'Last',range(3,6),
                              1,'First',range(3,6))



    # Enforce continuity between the last time in phase1 (time is index 7)
    # And the first time in phase2 (time is index 6!!)
    ocp.addDirectLinkEqualCon(1,'Last',[7],
                              2,'First',[6])


    # Enforce that the ODE parameters in phase 0 and phase 1 are equal
    ocp.addDirectLinkEqualCon(0,'ODEParams',[0],
                              1,'ODEParams',[0])

Another common case is when we have a list of phases all sequentially ordered in time and want to enforce forward time continuity in some common
set of state, time, or control variables.
This could be accomplished using :code:`addDirectLinkEqualCon` in a loop to link the :code:`'Last'` and :code:`'First'` states of
each adjacent phase as shown below. Alternatively, you can use the convenient :code:`addForwardLinkEqualCon` to accomplish the same thing.

.. code-block:: python
    
    # Enforce forward time continuity in XtUvars [0,1,2] across all phases
    for i in range(0,5):
        ocp.addDirectLinkEqualCon(i,'Last',range(0,3),
                                  i+1,'First',range(0,3))

    ## These accomplish the same thing
    ocp.addForwardLinkEqualCon(0,5,range(3,6))
    ###
    ocp.addForwardLinkEqualCon(phase0,phase5,range(3,6))


We should note that basically all simple continuity constraints between phases can be 
implemented using a combination of :code:`addDirectLinkEqualCon` and :code:`addForwardLinkEqualCon`, 
and users should only have to fall back on the more general form in special cases.




Link Inequality Constraints
---------------------------
A link inequality constraint of the form :math:`\vec{g}(\vec{x}) \leq \vec{0}` can be added to the :code:`ocp` using the 
:code:`.addLinkInequalCon` method. The interface for two phase and multi-phase linking works exactly the same as :code:`addLinkEqualCon` in regards
to order of arguments the different calling signatures.

.. code-block:: python

    def ALinkInequalCon():
        VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
        return VS0.dot(VP1)


    XtUvars0 = [3,4,5]
    SPvars0  = [0]

    XtUvars1 = [3,1,2]
    OPvars1 = [0]

    ## Enforce that the dot procuct of the specified variables from each phase region <0
    ocp.addLinkInequalCon(ALinkInequalCon(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])

    


    SomeFunc = Args(6).head(3).dot(Args(6).tail(3))
    ## Only need XtUVars from phases 2 and 3 at the specified regions
    ocp.addLinkInequalCon(SomeFunc,
                        phase2,'Last', range(0,3),
                        phase3,'First',range(0,3))


    def TriplePhaseInequality():
        X0,X1,X2 = Args(9).tolist([(0,3),(3,3),(6,3)])
    
        return vf.sum(X0,X1,X2)


    XtUvars = range(3,6)
    SPvars = []  # none needed, leave empty but still pass it in
    OPvars = []  # none needed, leave empty but still pass it in
    LPvars = []  # none needed, leave empty but still pass it in

    # List of tuples of the variables and regions needed from each phase
    ocp.addLinkInequalCon(TriplePhaseInequality(),
                        [(3,'First', XtUvars,OPvars,OPvars),
                         (4,'First', XtUvars,OPvars,OPvars),
                         (5,'First', XtUvars,OPvars,OPvars)],
                        LPvars)

Additionally, similar to before, if you need to write an inequality constraint only in terms in the link parameters you can use the
:code:`addLinkParamInequalCon` method.

.. code-block:: python

    # Enforce that the norm of first 3 link params is < 1
    LPvec = [0,1,2]
    ocp.addLinkParamInequalCon(Args(3).norm()-1.0,LPvec)

    # Apply same constraint to multiple groups of 3 link params
    LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
    ocp.addLinkParamInequalCon(Args(3).norm()-1.0,LPvecs)

    
Link Objectives
---------------
You can also add objectives of the form :math:`f(x)` using the :code:`addLinkObjective` method. Once again, the calling signature 
for all methods is identical to :code:`addLinkEqualCon`. Likewise, objectives involving only the link parameters can be added with
:code:`addLinkParamObjective`. Otherwise, the only difference is that the objective must be an ASSET ScalarFunction.
As was the case with phase, if multiple link objectives are added to an :code:`ocp`, they along with any objectives defined in each
phase are implicitly summed by the optimizer.

.. code-block:: python

    def ALinkObjective():
        VS0,VP1 = Args(8).tolist([(0,4),(4,4)])
        return VS0.dot(VP1)  # is a scalar function


    XtUvars0 = [3,4,5]
    SPvars0  = [0]

    XtUvars1 = [3,1,2]
    OPvars1 = [0]


    ocp.addLinkObjective(ALinkObjective(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])



    # Minimize the sum of the norms of these groups of 3 link params

    LPvecs = [[0,1,2] ,[3,4,5],[6,7,8]]
    ocp.addLinkParamObjective(Args(3).norm(),LPvecs)



Solving and Optimizing
======================

After constructing an :code:`OptimalControlProblem` and its constituent phases, we can now use PSIOPT 
to solve or optimize the entire trajectory. As with phase, the settings of the optimizer can be manipulated through a 
reference to PSIOPT attached to the :code:`ocp` object. 
Additionally, as before, calls to the optimizer are handled through the :code:`ocp` itself as shown below. 
Both of these topics are handled in more details in the section on :ref:`PSIOPT <psiopt-guide>`.


.. code-block:: python

    ocp.optimizer ## reference to this ocps instance of psiopt
    ocp.optimizer.set_OptLSMode("L1")

    ## Solve just the dynamics,equality, and inequality constraints
    flag = ocp.solve()

    ## Optimize objective subject to the dynamic,equality, and inequality constraints
    flag = ocp.optimize()

    ## Call solve to find feasible point, then optimize objective subject to the dynamic,equality, and inequality constraints
    flag = ocp.solve_optimize()

    ## Same as above but calls solve if the optimize call fails to fully converge
    flag = ocp.solve_optimize_solve()

After finding a solution, you can retrieve the converged trajectories for each phase using either the original defined phases or
:code:`ocp.Phase(i)`. Additionally you can retrieve the values of any link parameters using :code:`returnLinkParams`.

.. code-block:: python

    Traj0 = ocp.Phase(0).returnTraj()
    Traj0 = phase0.returnTraj()   # Remember phase0 and ocp.Phase(0) are the same object!!

    StatParams1 = ocp.Phase(1).returnStaticParams()

    LinkParams = ocp.returnLinkParams()

Finally, you can refine the meshes for some or all of the constituent phases and then resolve the problem.

.. code-block:: python
    
    ocp.solve_optimize()
   
    for phase in ocp.Phases:
        phase.refineTrajManual(5000)

    ocp.optimize()


    Trajs = [phase.returnTraj() for phase in ocp.Phases]

    CTrajs = [phase.returnCostateTraj() for phase in ocp.Phases]








Miscellaneous Topics
====================

Referencing and Removing Constraints
------------------------------------

When adding any of the 3 types of constraints/objectives covered :ref:`previously <link-guide>`, an integer identifier or list of integers is returned by the 
method. This identifier can be used to remove a constraint/objective from the optimal control problem. This can be quite useful when you
want to express some homotopic or continuation scheme without having to create a new ocp at each step. Given the identifier for
a function of a certain type, it can be removed from the phase using the corresponding :code:`.removeLink#####(id)` method as shown below.

.. code-block:: python

    ###########################
    ## Ex. Equality Constraint

    eq1 = ocp.addLinkEqualCon(SomeFunc,
                        2,'Last', range(0,3),
                        3,'First',range(0,3))

    ocp.removeLinkEqualCon(eq1)

    ###########################
    ## Ex. Inequality Constraint

    iq1 = ocp.addLinkInequalCon(ALinkInequalCon(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])


    ocp.removeLinkInequalCon(iq1)

    ###########################
    ## Ex.Link Objective

    ob1 = ocp.addLinkObjective(ALinkObjective(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])

    ocp.removeLinkObjective(ob1)


Retrieving Constraint Violations and Multipliers
------------------------------------------------
Immediately after a call to PSIOPT, users can retrieve the constraint violations and Lagrange multipliers associated with user applied link constraints.
For equality and inequality constraints, constraint violations and multipliers are retrieved by supplying a constraint function's id to the phase's :code:`.returnLink####Vals(id)` and :code:`.returnLink####Lmults(id)` methods as shown below.
In all cases, the violations/multipliers are returned as a list of numpy arrays each of which contains the output/multipliers associated with each call to the function inside
the optimization problem.

.. code-block:: python

    
    eq1 = ocp.addLinkEqualCon(SomeFunc,
                        2,'Last', range(0,3),
                        3,'First',range(0,3))

   
    iq1 = ocp.addLinkInequalCon(ALinkInequalCon(),
                        0,'Last', XtUvars0,[],SPvars0,
                        1,'First',XtUvars1,OPvars1,[])

    ocp.optimize()

    ecvals = ocp.returnEqualConVals(eq1)
    ecmults = ocp.returnEqualConLmults(eq1)

    icvals  = ocp.returnInequalConVals(iq1)
    icmults = ocp.returnInequalConLmults(iq1)



Additionally, the multipliers and constraint values for the phases inside of an optimal control can be retrieved as shown in the :ref:`phase tutorial <phaseremove-guide>`.