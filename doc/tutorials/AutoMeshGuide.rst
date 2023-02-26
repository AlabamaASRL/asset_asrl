.. _mesh-guide:

=================================
Adaptive Mesh Refinement Tutorial
=================================

!!!!!!WIP!!!!!!!!

In version 0.1.0, we have added significantly improved capabilities for automatic mesh refinement for problems
defined through the phase and OptimalControlProblem interfaces. 

..  note:: 

    These new features are entirely opt-in, and any old code should work exactly as it did before. 


Mathematical Background
=======================


Given a mesh with N segments of order p, the algorithm first estimates of the maximum error e_i in the ith segment spanning the time interval [t_i,t_{i+1}].
We have implemented two methods to obtain these error estimates.

.. math::

    e_i \quad on \quad [t_i,t_{i+1}] \quad i = 1 \ldots N

The first, which we refer to as deboors method,  estimates the error from the p+1th derivative of the solution as shown below [1,2,3]. 

.. math::

    e_i = C max[\vec{X}(t)^{p+1}]*h_i^{p+1} \quad where \quad h_i = t_{i+1} -t_{i}

Since the solution is a piecewise polynomial of order p, X^p is calculated using the differing scheme described by deboor[1]. The error coefficient C associated with an LGL method
of order p can be calculated using the method described by Russell [2].
For the second method, we estimate e_i by reintegrating the solution between all collocation points within each segment, and calculating the average error between the
integrated states and the collocation solution.

For well behaved problems, and sufficient numbers of segments these error estimates agree very closely with one another. However, in certain circumstances one may be superior to the other.
For example, the integration method more accurately estimates the true error on a coarse mesh. However, for some stiff problems, explicit integration be extremely slow or worse, fail, while deboor's method will be unaffected.

Having calculated the error e_i in each segment with either of the two methods, we then estimate the number of segments that will reduce e_i below some user specified tolerance. This estimate
is obtained by summing up the fractional number of segments, that each individual segment of the initial mesh 
would need to be divided into in order to meet the tolerance \epsilon/\sigma. Here epsilon is the user defined
error tolerance, and \sigma is user defined factor that exaggerates the error in each segment. The user defined \kappa factor enforces a maximum reduction on the number of segments in the next mesh

.. math::

    ^+N = ceil\left( \sum_{1}^{N} \text{max}\left(  \left(\frac{\sigma*e_i}{\epsilon}\right)^{\frac{1}{p+1}}  ,\kappa\right) \right)

Next, we calculate a new mesh spacing with N+ time intervals with approximately equal error.
This is done by first constructing a peice-wise constant error distribution function E_i(t) from our previous mesh as shown below.

.. math::
    
    E(i) = \frac{e_i^{\frac{1}{p+1}}}{h_i} on [t_i,t_{i+1}];

We then integrate and normalize this error distribution to obtain a piece-wise linear cumulative error function \bar{I}(t)

.. math::

 \bar{I}(t) = \frac{I(t)}{I(t_{N+1})} \quad where \quad I(t) = \int_{t_1}^{t_{N+1}} E(t) dt

The new times are th

.. math::
    
    ^+t_i = \bar{I}^{-1}(\frac{i}{^+N + 1}) \quad i = 1 \ldots N 


Phase
=====


Optimal Control Problem 
=======================


Jet Interface
=============



References
##########
#. de Boor, C. (1973). Good approximation by splines with variable knots. In Spline Functions and Approximation Theory: Proceedings of the Symposium held at the University of Alberta, Edmonton May 29 to June 1, 1972 (pp. 57-72). Birkhäuser Basel.
#. Russell, R. D., & Christiansen, J. (1978). Adaptive mesh selection strategies for solving boundary value problems. SIAM Journal on Numerical Analysis, 15(1), 59-80.
#. T Ozimek, M., J Grebow, D., & C Howell, K. (2010). A collocation approach for computing solar sail lunar pole-sitter orbits. The Open Aerospace Engineering Journal, 3(1).