.. _mesh-guide:

=================================
Adaptive Mesh Refinement Tutorial
=================================

In version 0.1.0, we have added significantly improved capabilities for automatic mesh refinement for problems
defined through the phase and OptimalControlProblem interfaces. 

..  note:: 

    These new features are entirely opt-in, and any old code "should" work exactly as it did before. 


Mathematical Background
=======================


Given a mesh with :math:`N` LGL segments of order :math:`p`, the algorithm first estimates the maximum error :math:`e_i` in the ith segment spanning the time interval :math:`[t_i,t_{i+1}]`.
We have implemented two methods to obtain these error estimates.

.. math::

    e_i \quad \text{on} \quad [t_i,t_{i+1}] \quad i = 1 \ldots N

The first, which we refer to as de Boor's method,  estimates the error from the :math:`p+1` th derivative, :math:`\vec{X}^{'(p+1)}` , of the solution as shown below [1,2,3]. 

.. math::

    e_i = C*h_i^{p+1} * |\vec{X}^{'(p+1)}|_{\infty} \quad \text{where} \quad h_i = t_{i+1} -t_{i}

Since the solution is a piecewise polynomial of order :math:`p`, :math:`\vec{X}^{'(p+1)}` is calculated using the differencing scheme described by de Boor[1]. The error coefficient, :math:`C`, associated with an LGL method
of order :math:`p` can be calculated using the method described by Russell [2].

For the second error estimation scheme, we calculate :math:`e_i` by reintegrating the solution between all collocation points within each segment, and calculating the average error between the
integrated states and the collocation solution.

For well behaved problems, and sufficient numbers of segments, these error estimates agree well with one another. However, in certain circumstances one may be superior to the other.
For example, the integration method more accurately estimates the true error on a coarse mesh. However, for some stiff problems, explicit integration can be extremely slow or worse, fail, while de Boor's method will be unaffected.

Having calculated the error :math:`e_i` in each segment with either of the two methods, we then estimate the number of segments that will reduce :math:`e_i` below some user specified tolerance. This estimate
is obtained by summing up the fractional number of segments that each individual segment of the initial mesh 
would need to be divided into in order to meet the tolerance :math:`\frac{\epsilon}{\sigma}`. Here :math:`\epsilon` is the user defined
mesh error tolerance, and :math:`\sigma` is a user defined factor that exaggerates the error in each segment. The user defined :math:`\kappa` and :math:`\gamma` factors enforce a maximum reduction and increase in
the number of segments in the next mesh respectively. We have found that this sum of fractional segments works better for our purposes than using the average value of :math:`e_i` as is done in [3].

.. math::

    ^+N = \text{ceil}\left[ \text{min}\left[  \sum_{1}^{N} \text{max}\left(  \left(\frac{\sigma*e_i}{\epsilon}\right)^{\frac{1}{p+1}}  ,\kappa\right) ,\gamma*N\right]\right]

Next, we calculate a new mesh spacing with :math:`^+N` time intervals with approximately equal error.
This is done by first constructing a piece-wise constant error distribution function :math:`E(t)` from our previous mesh as shown below [3].

.. math::
    
    E(t) = \frac{e_i^{\frac{1}{p+1}}}{h_i} \quad \text{on} \quad [t_i,t_{i+1}];

We then integrate and normalize this error distribution to obtain a piece-wise linear cumulative error function :math:`\bar{I}(t)`

.. math::

 \bar{I}(t) = \frac{I(t)}{I(t_{N+1})} \quad \text{where} \quad I(t) = \int_{t_1}^{t_{N+1}} E(t) dt

The new mesh's times are then are then chosen by evaluating the functional inverse, :math:`\bar{I}^{-1}(x)`, of the cumulative error function on an evenly spaced
grid of :math:`^+N + 1` points on the interval :math:`[0,1]` [3].

.. math::
    
    ^+t_i = \bar{I}^{-1}\left(\frac{i}{^+N }\right) \quad i = 0 \ldots ^+N 


At this point, we construct a new mesh at these time points from our old one and resolve/optimize the trajectory. This entire process repeats until the maximum error :math:`e_i`
is reduced below :math:`\epsilon`.

..  note:: 

    Our formulation allows all of the mesh times to translate and dilate/expand as needed during solving/optimization, but the relative non-dimensional spacing of times is constrained to remain constant.


Phase
=====

To enable the new adaptive mesh algorithm on a phase you need to call the object's :code:`.setAdaptiveMesh` method prior to invoking the optimizer.

.. code:: python
    

      phase.setAdaptiveMesh(True)  #Enable Adaptive mesh for all following solve/optimize calls
      phase.setAdaptiveMesh()      #Equivalent to above

      #phase.setAdaptiveMesh(False) #Or Disable it if turned on

The error estimation method discussed in the previous section can be selected by calling the phases's :code:`.setMeshErrorEstimator` function.
Remember, when using the :code:`"integrator"` method, all integration will done using the :code:`.integrator` object attached to the phase. Therefore, if the default tolerances
and step sizes are inappropriate for your problem, you should modify them.

.. code:: python
    
    phase.setMeshErrorEstimator('deboor')     #default
    phase.setMeshErrorEstimator('integrator')

    # Make sure the integrator is configured correctly for you problem
    phase.integrator.setAbsTol(1.0e-10)        # Recall,defaults to 1.0e-12
    phase.integrator.setRelTol(1.0e-12)        # Recall,defaults to 0.0
    phase.integrator.setStepSizes(.1,.001,1)   # Recall,defaults to .1,.1/10000,.1*10000


The mesh tolerance and max number of mesh iterations can be specified with the :code:`.setMeshTol` and :code:`.setMaxMeshIters` functions of the phase.
As a general rule of thumb, you should set the optimizer's equality constraint tolerance to be the same as or smaller than the mesh tolerance.

.. code:: python
    
    ## Set Error tolerance on mesh (epsilon) 
    phase.setMeshTol(1.0e-7)  #default = 1.0e-6
    ## Make sure to set optimizer EContol to be the same as or smaller than MeshTol
    phase.optimizer.set_EContol(1.0e-7)
    
    ## Set Max number of mesh iterations: 
    phase.setMaxMeshIters(10)  #default = 10

The hyper parameters, :math:`\sigma`, :math:`\kappa` , and :math:`\gamma`  of the mesh refinement algorithm can be set as shown below.

.. code:: python

    ## (sigma) Mesh Error exaggeration factor 
    phase.setMeshErrFactor(10.0)  #default = 10

    ## (kappa) Minimum multiple by which the # of segments can be reduced between iterations 
    phase.setMeshRedFactor(.5)  #default = .5

    ## (gamma) Maximum multiple by which the # of segments can be increased between iterations
    phase.setMeshIncFactor(5.0)  # default = 5
    
    phase.setMinSegments(4)      # default = 4
    phase.setMaxSegments(10000)  # default = 10000

Finally, you may also change the criteria used to determine whether the mesh has converged. By default, we consider the mesh converged when
:math:`\text{max}[e_1, \ldots,e_N] <\epsilon`. However, you can loosen this to converge when the time weighted average value of all :math:`e_i` satisfies the tolerance.

.. math::
    
    \left( \sum_1^N e_i h_i\right) \frac{1}{t_{N+1}-t_1} < \epsilon

Alternatively, you can set the convergence criteria to be the maximum error between the terminal state of the collocation solution
and one calculated by explicitly integrating the initial state and entire control history from the beginning to the end of the trajectory. 

These mesh error criteria may be set as shown below. As with the integrator based local error estimator, for :code:`'endtoend'` the phases integrator instance will be used to
reintegrate the control history, so modify tolerances and step sizes accordingly. Additionally, since the end to end error estimate is decoupled from the 
per segment error estimates used to generate the new mesh, users should be more aggressive with the mesh error exaggeration factor.

.. code:: python

    phase.setMeshErrorCriteria('max')  # default
    phase.setMeshErrorCriteria('avg')
    phase.setMeshErrorCriteria('endtoend')

    # If endtoend you might want to increase this parameter
    #phase.setMeshErrFactor(50.0) 

Finally, having specified all relevant parameters, we can solve or optimize the phase as we normally would. However, now
at each mesh iteration, additional information (see figure) pertaining to the progress of the refinement process will be printed along with the normal optimizer output.
As in the non-adaptive case, the flag returned by the call is the convergence flag of the last call made to PSIOPT. It does not indicate whether the mesh meets the error
tolerances. That is checked by reading the read-only :code:`.MeshConverged` field of the phase. 

..  note:: 

    If adaptive mesh refinement has not been enabled, .MeshConverged has no meaning and will be false!! 


.. code:: python

    # (Optional) Suppress optimizer output to only convergence status
    phase.optimizer.PrintLevel = 2

    # Enable or disable printing mesh info
    phase.PrintMeshInfo = True

    flag = phase.optimize()

    if(phase.MeshConverged and flag==0):
        print("Mesh converged and optimal")
    elif(flag==0):
        print("Optimal but mesh not converged")
    elif(phase.MeshConverged):
        print("Mesh converged, but not optimal and may not satisfy all non-dynamic constraints")
    else:
        print("Try Again")


.. image:: _static/PhaseMeshPrint.PNG
    :width: 60%

Optimal Control Problem 
=======================

Adaptive mesh refinement can also be enabled for multi-phase :code:`OptimalControlProblem` objects as well.

.. code:: python

    ocp  = oc.OptimalControlProblem()

    ocp.addPhase(phase1)
    ocp.addPhase(phase2)
    ocp.addPhase(phase3)

    EnableAdaptive = True
    ApplyToAllPhases = True  
    
    ## Turn on adaptive mesh for the ocp and enable it for all phases CURRENTLY in the ocp
    ocp.setAdaptiveMesh(EnableAdaptive,ApplyToAllPhases)
    
    ## Turn it on for the ocp, but not the phases: allows you selectively enable
    ## which phases are going to have adaptive mesh
    ocp.setAdaptiveMesh(True,False)
    
    ocp.setAdaptiveMesh(False) # Equivalent to ocp.setAdaptiveMesh(False,False)
    ocp.setAdaptiveMesh(True)  # Equivalent to ocp.setAdaptiveMesh(True,True)
    ocp.setAdaptiveMesh()      # Equivalent to ocp.setAdaptiveMesh(True,True)
    

Similarly, we can apply uniform settings/parameters to the mesh refinement process on each phase as shown below.

.. code:: python
    
    ocp.setMeshTol(1.0e-7)
    ocp.setMeshErrorEstimator('integrator')
    ocp.setMeshErrorCriteria('endtoend')
    ocp.setMeshRedFactor(.5)
    ocp.setMeshErrFactor(10.0)
    # etc..

    ## Equivalent to
    for phase in ocp.Phases:
        phase.setMeshTol(1.0e-7)
        phase.setMeshErrorEstimator('integrator')
        phase.setMeshErrorCriteria('endtoend')
        phase.setMeshRedFactor(.5)
        phase.setMeshErrFactor(10.0)

Additionally, it is not required that all phases have the same tolerances or parameters. You may set them
each individually, or even disable adaptive mesh refinement on some phases as well.

.. code:: python

    ocp.Phase(0).setMeshErrorEstimator('endtoend')
    ocp.Phase(1).setMeshTol(1.0e-9)
    ocp.Phase(2).setAdaptiveMesh(False)

    



However,the maximum number of mesh iterations and the console printing are controlled only by the :code:`ocp` object.

.. code:: python

    ocp.setMaxMeshIters(10)
    ocp.PrintMeshInfo = True

    # fine, but ignored
    ocp.Phase(0).setMaxMeshIters(10) 
    ocp.Phase(0).PrintMeshInfo = True


With adaptive mesh refinement enabled, we continually solve/optimize, the entire problem until 
all constituent phases with adaptive mesh refinement enabled have converged or the maximum number of mesh iterates is reached.
Once any one of the phases is converged, we do not modify the mesh spacing or number of segments of that phase
on subsequent mesh iterations so long as it continues to satisfy the error tolerances.
At each mesh iteration, additional information (see figure) pertaining to the progress of the refinement process will be printed along with the normal optimizer output.
The convergence status of all phases (with adaptive mesh refinement enabled) can be checked using the :code:`.MeshConverged` field of the :code:`ocp`. Alternatively
you can also query the convergence status of the individual phases themselves. 

.. code:: python

    
    # (Optional) Suppress optimizer output to only on print convergence status
    phase.optimizer.PrintLevel = 2

    flag = ocp.optimize()

    if(ocp.MeshConverged and flag==0):
        print("Problem solved to optimality and all phase meshes converged")
    
    else:
        for i,phase in enumerate(ocp.Phases):
            print(f"Phase {i} Converged??:",phase.MeshConverged)


.. image:: _static/OcpMeshPrint.PNG
    :width: 60%





References
##########
#. de Boor, C. (1973). Good approximation by splines with variable knots. In Spline Functions and Approximation Theory: Proceedings of the Symposium held at the University of Alberta, Edmonton May 29 to June 1, 1972 (pp. 57-72). Birkhauser Basel.
#. Russell, R. D., & Christiansen, J. (1978). Adaptive mesh selection strategies for solving boundary value problems. SIAM Journal on Numerical Analysis, 15(1), 59-80.
#. T Ozimek, M., J Grebow, D., & C Howell, K. (2010). A collocation approach for computing solar sail lunar pole-sitter orbits. The Open Aerospace Engineering Journal, 3(1).