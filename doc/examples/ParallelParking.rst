Parallel Parking
=============================

.. _example-parallel-parking:

Overview
--------

This example demonstrates a minimum-time parallel parking maneuver formulated
and solved using the ASSET optimal control framework.

The vehicle is modeled using a kinematic bicycle model with acceleration and
steering dynamics. The optimization determines the control inputs required to
move the vehicle from its initial position on the roadway into a parallel
parking space while satisfying vehicle dynamics, control limits, roadway
boundaries, parking-space constraints, and collision-avoidance constraints.

The problem is derived from the ICLOCS parallel parking example:

`ICLOCS Parallel Parking Example
<http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html>`_.

The ICLOCS example is based on Case 7 in Table 2 of the following research
paper:

`IEEE Reference Paper <https://ieeexplore.ieee.org/document/7463491>`_.

The initial conditions used in this implementation follow the referenced
research paper. The resulting maneuver time is compared against the reported
reference value of 18.426 seconds.

Problem Formulation
-------------------

The objective is to minimize the total maneuver time:

.. math::

   \min_{t_f} \quad t_f

subject to the vehicle equations of motion, initial and terminal conditions,
state and control bounds, and geometric constraints.

The optimization is implemented using an ASSET optimal control phase with
Legendre-Gauss-Lobatto discretization.

Vehicle Dynamics
----------------

The vehicle is represented by a kinematic bicycle model. The state vector is

.. math::

   \mathbf{x} =
   \begin{bmatrix}
   x & y & v & a & \theta & \phi
   \end{bmatrix}^{T},

where:

- :math:`x` and :math:`y` are the vehicle rear-axle position coordinates.
- :math:`v` is the longitudinal velocity.
- :math:`a` is the longitudinal acceleration.
- :math:`\theta` is the vehicle heading angle.
- :math:`\phi` is the steering angle.

The control vector is

.. math::

   \mathbf{u} =
   \begin{bmatrix}
   j & \dot{\phi}
   \end{bmatrix}^{T},

where :math:`j` is longitudinal jerk and :math:`\dot{\phi}` is the steering
rate.

The equations of motion are

.. math::

   \dot{x} &= v\cos(\theta), \\
   \dot{y} &= v\sin(\theta), \\
   \dot{v} &= a, \\
   \dot{a} &= j, \\
   \dot{\theta} &= \frac{v\tan(\phi)}{L}, \\
   \dot{\phi} &= \dot{\phi},

where :math:`L` is the wheelbase.

Vehicle Geometry
----------------

The vehicle geometry is defined relative to the rear axle. The four vehicle
corners are used to formulate roadway, parking-space, and collision-avoidance
constraints.

The vehicle dimensions used in the example are:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Parameter
     - Symbol
     - Value
   * - Front overhang
     - :math:`l_f`
     - 0.839 m
   * - Wheelbase
     - :math:`L`
     - 2.588 m
   * - Rear overhang
     - :math:`l_r`
     - 0.657 m
   * - Half vehicle width
     - :math:`b`
     - 0.8855 m

Parking-Space Geometry
----------------------

The parking space and roadway are represented using geometric boundary
constraints.

The parking-space dimensions are:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Parameter
     - Symbol
     - Value
   * - Parking-space length
     - :math:`L_s`
     - 5.0 m
   * - Parking-space width
     - :math:`W_s`
     - 2.0 m
   * - Roadway boundary
     - :math:`W_r`
     - 3.5 m

A smoothed Heaviside function is used to represent the parking-space boundary:

.. math::

   H(x,k) = \frac{1+\tanh(kx)}{2},

where :math:`k` controls the smoothing of the transition.

The lower parking-space boundary is defined as

.. math::

   y_s(x) =
   \left[-H(x,k)+H(x-L_s,k)\right]W_s.

This formulation provides a smooth representation of the parking-space
entrance and exit boundaries.

Initial Conditions
------------------

The initial vehicle position and heading follow the reference research paper.

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Parameter
     - Symbol
     - Value
   * - Initial horizontal position
     - :math:`x_0`
     - -5.14 m
   * - Initial vertical position
     - :math:`y_0`
     - 1.41 m
   * - Initial heading
     - :math:`\theta_0`
     - 13.18 degrees
   * - Initial velocity
     - :math:`v_0`
     - 0 m/s
   * - Initial acceleration
     - :math:`a_0`
     - 0 m/s\ :sup:`2`
   * - Initial steering angle
     - :math:`\phi_0`
     - 0 rad

The initial trajectory guess is constructed using a sequence of intermediate
vehicle positions and heading angles. This guess provides an initial maneuver
for the nonlinear optimization.

Constraints
-----------

The optimization includes the following constraints.

Vehicle dynamics
~~~~~~~~~~~~~~~~

The vehicle state evolves according to the kinematic bicycle model.

Initial conditions
~~~~~~~~~~~~~~~~~~

The initial vehicle state is fixed using an ASSET boundary-value constraint.

Terminal conditions
~~~~~~~~~~~~~~~~~~~

The final velocity and acceleration are constrained to zero:

.. math::

   v(t_f) = 0, \qquad a(t_f) = 0.

The four vehicle corners are also constrained to lie within the parking-space
width at the final time.

Roadway and parking-space boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The vehicle corners must remain within the roadway and parking-space
boundaries throughout the maneuver. These constraints are evaluated using
the transformed vehicle corner coordinates.

Collision avoidance
~~~~~~~~~~~~~~~~~~~

Additional geometric constraints prevent the vehicle from intersecting the
two upper corners of the parking-space entrance. The constraints use triangle
areas formed by the vehicle corners and the parking-space entrance corners.

State and control bounds
~~~~~~~~~~~~~~~~~~~~~~~~

The following limits are imposed:

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Quantity
     - Symbol
     - Bound
   * - Steering angle
     - :math:`\phi`
     - :math:`\pm 33` degrees
   * - Velocity
     - :math:`v`
     - :math:`\pm 2.0` m/s
   * - Acceleration
     - :math:`a`
     - :math:`\pm 0.75` m/s\ :sup:`2`
   * - Jerk
     - :math:`j`
     - :math:`\pm 0.5` m/s\ :sup:`3`
   * - Curvature rate
     - :math:`\dot{\kappa}`
     - :math:`\pm 0.6`

The curvature rate is calculated from the steering angle and steering rate:

.. math::

   \dot{\kappa}
   =
   \frac{\dot{\phi}}{L\cos^2(\phi)}.

Optimization Setup
------------------

The problem is implemented using an ASSET ``ODEBase`` model and an optimal
control phase.

The initial optimization uses:

- LGL5 transcription.
- 50 mesh segments.
- Block-constant controls.
- Initial boundary smoothing parameter of 75.
- Maximum of 2000 optimizer iterations.
- Bound fraction of 0.995.

After the first optimization, the trajectory is refined to 200 mesh segments.
The smoothing parameter is increased to 150, and the refined problem is
optimized using a KKT tolerance of :math:`10^{-8}`.

The maneuver time is minimized using ASSET's delta-time objective.

Trajectory Reintegration
------------------------

After optimization, the resulting trajectory is independently reintegrated
using the vehicle model.

The reintegration uses a step size of 0.1 seconds and generates 500 dense
trajectory points. The optimized and reintegrated terminal states are
compared to assess trajectory consistency.

The terminal-state difference is calculated as

.. math::

   \Delta \mathbf{x}_f =
   \mathbf{x}_{f,\mathrm{optimized}} -
   \mathbf{x}_{f,\mathrm{reintegrated}}.

The norm of this difference is also reported.

Results
-------

The optimized maneuver time is compared with the reference maneuver time of
18.426 seconds.

The implementation reports:

- Optimized ASSET maneuver time.
- Reference maneuver time.
- Absolute maneuver-time difference.
- Relative maneuver-time difference.
- Optimized versus reintegrated terminal-state difference.
- Norm of the terminal-state difference.

The relative maneuver-time difference is calculated as

.. math::

   \Delta t_{\%}
   =
   100
   \frac{t_{\mathrm{ASSET}}-t_{\mathrm{reference}}}
        {t_{\mathrm{reference}}}.

The numerical results are printed when the example is executed. No numerical
optimization result is assumed in this documentation.

Trajectory Visualization
------------------------

The example generates a plot of the initial trajectory guess and the
optimized trajectory.

The vehicle is represented by its rectangular body, four wheels, and front
and rear axle lines. The parking-space boundary and roadway are plotted
alongside the vehicle trajectory.

The optimized trajectory is also independently reintegrated and animated.
The animation displays the vehicle motion, the trajectory history, and the
current simulation time.

Running the Example
-------------------

Run the Python example from the repository root:

.. code-block:: bash

   python examples/ParallelParking.py

The example requires ASSET, NumPy, and Matplotlib. Saving the animation as an
MP4 file additionally requires FFmpeg.

Full Code
---------

The complete Python implementation is included below.

.. literalinclude:: ../../examples/ParallelParking.py
   :language: python
   :linenos:
   :caption: ParallelParking.py

References
----------

.. [1] Imperial College London, "ICLOCS Parallel Parking Example."
   http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html

.. [2] IEEE, "Research paper containing the parallel parking Case 7
   formulation, Table 2."
   https://ieeexplore.ieee.org/document/7463491
