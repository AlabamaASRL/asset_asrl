===========================
Minimum-Time Parallel Parking
===========================

In this example, we solve a minimum-time parallel parking problem using
ASSET's optimal control framework. The problem is derived from the ICLOCS
parallel parking example [1], which is based on Case 7 in Table 2 of the
referenced research paper [2].

The vehicle is modeled using a kinematic bicycle model. The optimization
determines the vehicle motion and control inputs required to move from an
initial position on the roadway into a parallel parking space while satisfying
vehicle dynamics, state and control bounds, roadway boundaries, and
collision-avoidance constraints.

The initial conditions used in this implementation follow the referenced
research paper. The resulting maneuver time is compared with the reported
reference value of 18.426 seconds.

Problem Setup
#############

The parking space and roadway dimensions are defined below.

.. math::

    L_s = 5.0\;m \quad\quad W_s = 2.0\;m \quad\quad W_r = 3.5\;m

where :math:`L_s` is the parking-space length, :math:`W_s` is the
parking-space width, and :math:`W_r` is the roadway boundary.

The vehicle dimensions are:

.. math::

    l_f = 0.839\;m \quad\quad L = 2.588\;m
    \quad\quad l_r = 0.657\;m \quad\quad b = 0.8855\;m

where :math:`l_f` is the front overhang, :math:`L` is the wheelbase,
:math:`l_r` is the rear overhang, and :math:`b` is the vehicle half-width.

Vehicle Dynamics
################

The vehicle state vector is

.. math::

    \vec{X} =
    [x,y,v,a,\theta,\phi]^T

where :math:`x` and :math:`y` are the rear-axle position coordinates,
:math:`v` is the longitudinal velocity, :math:`a` is the longitudinal
acceleration, :math:`\theta` is the vehicle heading angle, and :math:`\phi`
is the steering angle.

The control vector is

.. math::

    \vec{U} = [j,\dot{\phi}]^T

where :math:`j` is longitudinal jerk and :math:`\dot{\phi}` is the steering
rate.

The equations of motion are

.. math::

    \dot{x} &= v\cos(\theta), \\
    \dot{y} &= v\sin(\theta), \\
    \dot{v} &= a, \\
    \dot{a} &= j, \\
    \dot{\theta} &= \frac{v\tan(\phi)}{L}, \\
    \dot{\phi} &= \dot{\phi}.

The vehicle is implemented as an ASSET ``ODEBase`` model. The four vehicle
corners are defined in the body-fixed frame and transformed into the inertial
frame using the vehicle heading angle.

Parking-Space Boundary
######################

A smoothed Heaviside function is used to represent the parking-space boundary:

.. math::

    H(x,k) = \frac{1+\tanh(kx)}{2}

where :math:`k` controls the smoothing of the transition.

The lower parking-space boundary is defined as

.. math::

    y_s(x) =
    \left[-H(x,k)+H(x-L_s,k)\right]W_s.

This provides a smooth representation of the parking-space entrance and exit
boundaries.

Initial Conditions
##################

The initial position and heading follow the referenced research paper.

.. math::

    x(0) = -5.14\;m \quad\quad y(0) = 1.41\;m

    \theta(0) = 13.18^\circ

    v(0) = 0 \quad\quad a(0) = 0 \quad\quad \phi(0) = 0

The initial trajectory guess is constructed using a sequence of intermediate
vehicle positions, heading angles, and time values.

Constraints
###########

The optimization enforces the following constraints:

- The initial vehicle state is fixed.
- The final vehicle velocity and acceleration are zero.
- All vehicle corners remain within the roadway and parking-space boundaries.
- The final vehicle corners lie within the parking-space width.
- Collision-avoidance constraints prevent intersection with the two upper
  parking-space entrance corners.
- Vehicle state and control variables remain within their specified bounds.
- The vehicle curvature rate remains bounded.

The state and control limits are

.. math::

    |\phi| \leq 33^\circ

    |v| \leq 2.0\;m/s

    |a| \leq 0.75\;m/s^2

    |j| \leq 0.5\;m/s^3

    |\dot{\kappa}| \leq 0.6

The curvature rate is calculated as

.. math::

    \dot{\kappa} =
    \frac{\dot{\phi}}{L\cos^2(\phi)}.

Optimization
############

The minimum-time maneuver is obtained by minimizing the final time:

.. math::

    J = t_f

The problem is transcribed using ASSET's LGL5 discretization and block-constant
control parameterization.

The initial optimization uses 50 mesh segments and a smoothing parameter of
75. After the first optimization, the trajectory is refined to 200 segments
and the smoothing parameter is increased to 150.

The optimizer is configured with a maximum of 2000 iterations and a bound
fraction of 0.995. The refined optimization uses a KKT tolerance of
:math:`10^{-8}`.

Trajectory Reintegration
########################

After optimization, the trajectory is independently reintegrated using the
vehicle model. The integration step size is 0.1 seconds, and 500 dense
trajectory points are generated.

The optimized and reintegrated terminal states are compared using

.. math::

    \Delta \vec{X}_f =
    \vec{X}_{f,\mathrm{optimized}} -
    \vec{X}_{f,\mathrm{reintegrated}}.

The norm of the terminal-state difference is also reported.

Results
#######

The optimized maneuver time is compared against the reference maneuver time
of 18.426 seconds.

The relative maneuver-time difference is calculated as

.. math::

    \Delta t_{\%} =
    100\frac{t_{\mathrm{ASSET}}-t_{\mathrm{reference}}}
    {t_{\mathrm{reference}}}.

The example prints the following results:

- ASSET maneuver time.
- Reference maneuver time.
- Absolute maneuver-time difference.
- Relative maneuver-time difference.
- Optimized versus reintegrated terminal-state difference.
- Norm of the terminal-state difference.

The numerical optimization results are generated when the Python example is
executed.

Trajectory Visualization
########################

The example plots the initial trajectory guess and the optimized vehicle
trajectory. The vehicle body, wheels, and axle locations are displayed along
with the roadway and parking-space boundaries.

An animation of the independently reintegrated trajectory is also generated.
The animation displays the vehicle motion, trajectory history, and current
simulation time.

Running the Example
###################

Run the Python example from the repository root:

.. code-block:: bash

    python examples/ParallelParking.py

The example requires ASSET, NumPy, and Matplotlib. FFmpeg is additionally
required if the animation is saved as an MP4 file.

Full Code
#########

The complete Python implementation is included below.

.. literalinclude:: ../../examples/ParallelParking.py
    :language: python
    :linenos:
    :caption: ParallelParking.py

References
##########

.. [1] Imperial College London, "ICLOCS Parallel Parking Example."
   http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html

.. [2] IEEE, "Research paper containing the parallel parking Case 7
   formulation, Table 2."
   https://ieeexplore.ieee.org/document/7463491
