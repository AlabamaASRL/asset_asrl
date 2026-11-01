Parallel Parking
================

This example demonstrates the use of ASSET's optimal control framework to solve a minimum-time parallel parking problem. A vehicle is maneuvered into a parking space with dimensions only marginally larger than the vehicle itself. The optimization accounts for vehicle dynamics, steering and acceleration limits, roadway boundaries, and collision avoidance.

The example is derived from the parallel parking problem presented in the `ICLOCS parallel parking example <http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html>`_, which is based on Case 7 in Table 2 of the research paper:

`Optimal Control for Automated Parallel Parking <https://ieeexplore.ieee.org/document/7463491>`_.

The initial conditions used here follow the referenced research paper. The resulting maneuver time is compared with the reported paper value of 18.426 seconds.

Problem Description
-------------------

The objective is to find a minimum-time vehicle trajectory that moves a car from an initial position on the roadway into a constrained parallel parking space.

The vehicle must satisfy the following requirements:

* Begin at a prescribed initial position and orientation.
* Remain within the roadway and parking-space boundaries.
* Avoid collisions with the corners of the parking-space entrance.
* Satisfy velocity, acceleration, steering-angle, jerk, and steering-rate limits.
* Finish with the vehicle positioned inside the parking space and with zero terminal velocity and acceleration.

The optimization uses a fixed initial state and a free final time. The parking-space geometry is represented using a smoothed step function to provide differentiable boundary constraints.

Vehicle Dynamics
----------------

The vehicle is modeled as a six-state kinematic bicycle model. The state vector is

.. math::

   \mathbf{x} =
   \begin{bmatrix}
   x & y & v & a & \theta & \phi
   \end{bmatrix}^{T},

where:

* :math:`x,y` are the vehicle position coordinates.
* :math:`v` is the longitudinal velocity.
* :math:`a` is the longitudinal acceleration.
* :math:`\theta` is the vehicle heading angle.
* :math:`\phi` is the steering angle.

The control vector contains the jerk and steering-angle rate:

.. math::

   \mathbf{u} =
   \begin{bmatrix}
   u_1 & u_2
   \end{bmatrix}^{T}.

The equations of motion are

.. math::

   \dot{x} &= v\cos(\theta), \\
   \dot{y} &= v\sin(\theta), \\
   \dot{v} &= a, \\
   \dot{a} &= u_1, \\
   \dot{\theta} &= \frac{v\tan(\phi)}{l_{\mathrm{axes}}}, \\
   \dot{\phi} &= u_2,

where :math:`l_{\mathrm{axes}}` is the distance between the front and rear axles.

The model is implemented by subclassing ``oc.ODEBase``. This allows the vehicle dynamics and geometric constraint functions to be used directly in ASSET's optimal control phase.

Vehicle Geometry
----------------

The vehicle is represented as a rectangular body defined by four corners in the vehicle body frame. The corner coordinates are measured relative to the center of the rear axle.

The vehicle dimensions are specified by:

* ``l_front``: distance from the front axle to the front of the vehicle.
* ``l_axes``: distance between the front and rear axles.
* ``l_rear``: distance from the rear axle to the rear of the vehicle.
* ``b_width``: half-width of the vehicle.

The body-frame corner coordinates are rotated into the inertial frame using the vehicle heading angle. These transformed coordinates are used to construct the roadway and parking-space constraints.

Roadway and Parking-Space Constraints
-------------------------------------

The parking space is defined using its length, width, and the roadway width. The roadway and parking-space boundaries are enforced at all four vehicle corners.

The parking-space boundary is modeled using a hyperbolic tangent approximation of the Heaviside step function:

.. math::

   H_k(x) = \frac{1+\tanh(kx)}{2},

where :math:`k` controls the sharpness of the transition. The parking-space boundary is constructed as

.. math::

   F_{\mathrm{slot}}(x)
   =
   \left[-H_k(x)+H_k(x-S_L)\right]S_W,

where :math:`S_L` is the parking-space length and :math:`S_W` is the parking-space width.

The smoothing parameter allows the boundary to be represented by a differentiable function, which is useful for gradient-based optimization. The parameter is initially set to a moderate value and increased during the refinement process to provide a closer approximation to the desired parking-space geometry.

Corner Collision Constraints
----------------------------

Additional constraints prevent the vehicle from colliding with the upper corners of the parking-space entrance.

These constraints use the areas of triangles formed by the vehicle corners and the two parking-space entrance corners. The sum of the triangle areas is compared with a reference area based on the vehicle dimensions.

The resulting inequalities constrain the vehicle geometry relative to both entrance corners. This provides an additional geometric collision-avoidance condition during the maneuver.

Terminal Constraints
--------------------

At the end of the maneuver, all four vehicle corners must lie within the parking-space width. The terminal longitudinal velocity and acceleration are also constrained to zero:

.. math::

   v(t_f)=0, \qquad a(t_f)=0.

Together, these conditions require the vehicle to finish inside the parking space in a stationary configuration.

Control and Path Constraints
----------------------------

The optimization enforces bounds on the vehicle states and controls. The example uses the following limits:

.. list-table::
   :header-rows: 1
   :widths: 35 30 30

   * - Quantity
     - Symbol
     - Bound
   * - Steering angle
     - :math:`\phi`
     - :math:`\pm 33^\circ`
   * - Velocity
     - :math:`v`
     - :math:`\pm 2\ \mathrm{m/s}`
   * - Acceleration
     - :math:`a`
     - :math:`\pm 0.75\ \mathrm{m/s^2}`
   * - Jerk
     - :math:`u_1`
     - :math:`\pm 0.5\ \mathrm{m/s^3}`
   * - Steering-angle rate
     - :math:`u_2`
     - Bounded through the curvature-rate constraint

The curvature-rate constraint is defined by

.. math::

   \dot{\kappa}
   =
   \frac{u_2}
   {l_{\mathrm{axes}}\cos^2(\phi)}.

The absolute value of this quantity is limited to 0.6 in the example.

Optimal Control Formulation
---------------------------

The problem is formulated as an ASSET optimal control phase using the ``LGL5`` transcription. The initial trajectory guess is constructed from a sequence of vehicle positions, headings, and times.

The initial guess provides a rough maneuver consisting of several forward and reverse motions. The optimizer then adjusts the trajectory and control history to satisfy the constraints while minimizing the maneuver time.

The objective is

.. math::

   \min_{\mathbf{x}(t),\mathbf{u}(t),t_f} \quad t_f.

The phase uses block-constant controls and a static smoothing parameter for the parking-space boundary.

The optimization proceeds in two stages:

#. The initial problem is solved using 50 segments and a smoothing parameter of 75.
#. The trajectory is refined to 200 segments, the smoothing parameter is increased to 150, and the problem is optimized again using a tighter KKT tolerance.

This refinement procedure improves the representation of the parking-space boundary while increasing the temporal resolution of the trajectory.

Trajectory Propagation and Validation
-------------------------------------

After optimization, the resulting trajectory is extracted from the phase and used to construct an integration table. The vehicle dynamics are then independently reintegrated using an ASSET integrator.

The reintegrated trajectory is sampled at 500 points. The terminal state difference between the optimized trajectory and the reintegrated solution is printed to assess the consistency of the trajectory and the vehicle dynamics.

The maneuver time is also compared against the reference value reported in the research paper:

.. math::

   t_{\mathrm{paper}} = 18.426\ \mathrm{s}.

Visualization and Animation
---------------------------

The example includes plotting and animation utilities to visualize the parking maneuver.

The trajectory visualization displays:

* The roadway and parking-space boundaries.
* The vehicle at selected points along the trajectory.
* The vehicle orientation and position throughout the maneuver.

The animation additionally displays the vehicle body, front and rear axles, wheels, and the path traced by the vehicle. The simulation time is displayed as the maneuver progresses.

The animation can optionally be saved as an MP4 file using Matplotlib's ``FFMpegWriter``. This requires FFmpeg to be installed and available on the system path.

Running the Example
-------------------

The example requires the following Python packages:

* NumPy
* Matplotlib
* Seaborn
* ASSET

Seaborn can be installed using:

.. code-block:: bash

   pip install seaborn

Run the example from the repository using:

.. code-block:: bash

   python asset_asrl/examples/ParallelParking.py

The main routine constructs the vehicle model, defines the optimal control phase, solves and refines the trajectory, reintegrates the solution, and displays the trajectory and animation.

Full Code
---------

The complete Python implementation is included directly from the ASSET repository. This ensures that the code displayed in the documentation remains synchronized with the executable example.

.. literalinclude:: ../../examples/ParallelParking.py
   :language: python
   :linenos:
   :caption: ParallelParking.py

References
----------

.. [1] Imperial College London, `ICLOCS Parallel Parking Example <http://www.ee.ic.ac.uk/ICLOCS/ExampleParallelParking.html>`_.

.. [2] `IEEE research paper, Document 7463491 <https://ieeexplore.ieee.org/document/7463491>`_, Case 7 in Table 2.
