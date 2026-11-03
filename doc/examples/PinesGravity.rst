Pines Gravity Model
===================

This example demonstrates the use of ASSET's native Pines gravity kernel for
evaluating high-fidelity gravitational acceleration and propagating orbital
trajectories. The example uses spherical harmonic gravity coefficients from
the EGM96 gravity model and compares a point-mass gravity solution against a
degree-8 spherical harmonic solution.

The Pines gravity implementation provides gravitational acceleration through
an ASSET VectorFunction interface. This allows the gravity model to be used
directly within ASSET dynamics and provides access to derivatives such as the
acceleration Jacobian.

The example demonstrates four primary capabilities:

#. Constructing Pines gravity models at different spherical harmonic degrees.
#. Evaluating gravitational acceleration and comparing it against point-mass gravity.
#. Computing derivatives of the gravitational acceleration VectorFunction.
#. Propagating orbital trajectories using point-mass and spherical harmonic gravity.

.. note::

   The example uses publicly available EGM96 gravity coefficients. The
   coefficient file is loaded from the repository's native data directory.
   The gravity model is evaluated in a fixed body-centered coordinate frame.
   Earth rotation and time-dependent transformations between inertial and
   body-fixed frames are not included in this example.

Gravity Model
-------------

The gravitational acceleration is evaluated using the Pines gravity kernel.
The kernel includes central gravity and the selected spherical harmonic
perturbations.

The gravitational model is parameterized by the gravitational parameter
:math:`\mu`, reference radius :math:`R_E`, maximum spherical harmonic degree
:math:`N`, and a set of fully normalized spherical harmonic coefficients.

The constants used in this example are

.. math::

   \begin{aligned}
   \mu &= 3.9860043543609598 \times 10^5
   \;\mathrm{km^3/s^2}, \\
   R_E &= 6378.1363 \;\mathrm{km}.
   \end{aligned}

The gravity coefficients are read from the EGM96 coefficient file.

The gravity model is constructed using :code:`astro.PinesGravity`, which
accepts the coefficient file, maximum degree, gravitational parameter,
reference radius, and coefficient normalization convention.

The :code:`lru_cache` decorator ensures that each requested gravity model is
constructed only once. This avoids repeatedly parsing the coefficient file
when the same harmonic degree is used for multiple acceleration evaluations
or propagations.

Gravity Evaluation
------------------

The first part of the example evaluates gravitational acceleration at a
single position. The selected position is

.. math::

   \mathbf{r} =
   \begin{bmatrix}
   4510 \\
   -3120 \\
   5230
   \end{bmatrix}
   \;\mathrm{km}.

For comparison, the point-mass gravitational acceleration is calculated
analytically using

.. math::

   \mathbf{a}_{2B}
   =
   -\frac{\mu}{r^3}\mathbf{r},

where

.. math::

   r = \|\mathbf{r}\|.

The Pines gravity kernel is evaluated at spherical harmonic degrees
0, 2, 8, and 20. Degree zero represents the central point-mass contribution,
while increasing the maximum degree includes additional spherical harmonic
terms.

The acceleration magnitude and difference from the analytical point-mass
solution provide a basic comparison of the gravity models. The difference
measures the perturbing contribution relative to central gravity at the
selected position.

Acceleration Derivatives
------------------------

In addition to evaluating acceleration, the Pines VectorFunction interface
can provide derivatives of the gravitational acceleration.

For a position vector :math:`\mathbf{r}`, the acceleration Jacobian is

.. math::

   \frac{\partial \mathbf{a}}{\partial \mathbf{r}}
   =
   \begin{bmatrix}
   \dfrac{\partial a_x}{\partial x} &
   \dfrac{\partial a_x}{\partial y} &
   \dfrac{\partial a_x}{\partial z} \\
   \dfrac{\partial a_y}{\partial x} &
   \dfrac{\partial a_y}{\partial y} &
   \dfrac{\partial a_y}{\partial z} \\
   \dfrac{\partial a_z}{\partial x} &
   \dfrac{\partial a_z}{\partial y} &
   \dfrac{\partial a_z}{\partial z}
   \end{bmatrix}.

This derivative describes how the gravitational acceleration changes with
position and is useful for sensitivity analysis, variational equations, and
gradient-based optimization.

The :code:`compute_jacobian` method returns both the acceleration and its
Jacobian with respect to the input position.

Orbital Dynamics
----------------

To demonstrate the use of the Pines gravity kernel in an ASSET propagation,
the gravitational acceleration is incorporated into a six-state orbital
dynamics model.

The state is defined as

.. math::

   \mathbf{x}
   =
   \begin{bmatrix}
   \mathbf{r} \\
   \mathbf{v}
   \end{bmatrix},

where :math:`\mathbf{r}` and :math:`\mathbf{v}` are the position and velocity
vectors, respectively.

The equations of motion are

.. math::

   \begin{aligned}
   \dot{\mathbf{r}} &= \mathbf{v}, \\
   \dot{\mathbf{v}} &= \mathbf{a}_{\mathrm{Pines}}(\mathbf{r}).
   \end{aligned}

The acceleration function includes central gravity and the selected
spherical harmonic perturbations.

The dynamics are implemented using :code:`oc.ODEBase`.

The gravity model is passed directly into the dynamics constructor. This
preserves the ASSET VectorFunction representation of the acceleration and
allows the resulting ODE to be integrated using ASSET's numerical integrators.

Initial Conditions
------------------

The propagation begins with a circular orbit at an initial radius of
:math:`7000` km. The orbit is inclined by :math:`51.6^\circ`, representative
of an inclined low Earth orbit.

For a circular point-mass orbit, the initial velocity magnitude is

.. math::

   v_0 = \sqrt{\frac{\mu}{r_0}}.

The initial position and velocity are

.. math::

   \begin{aligned}
   \mathbf{r}_0 &=
   \begin{bmatrix}
   7000 \\
   0 \\
   0
   \end{bmatrix}
   \;\mathrm{km}, \\[1em]
   \mathbf{v}_0 &=
   \begin{bmatrix}
   0 \\
   v_0\cos(51.6^\circ) \\
   v_0\sin(51.6^\circ)
   \end{bmatrix}
   \;\mathrm{km/s}.
   \end{aligned}

The initial state contains seven entries because the integrator's trajectory
output includes time as an additional variable. The ODE itself contains six
state variables.

The orbital period of the corresponding point-mass circular orbit is

.. math::

   T = 2\pi\sqrt{\frac{r_0^3}{\mu}}.

The propagation duration is selected as ten orbital periods.

Orbit Propagation
-----------------

Two gravity models are propagated using the same initial conditions:

#. Degree 0: central point-mass gravity.
#. Degree 8: EGM96 spherical harmonic gravity through degree 8.

Both models use an ASSET integrator with a nominal integration step of
30 seconds and an absolute tolerance of :math:`10^{-8}`.

The resulting trajectories contain position, velocity, and time histories.
The two trajectories can therefore be compared at matching output times.

Point-Mass Propagation Check
----------------------------

A separate propagation using the degree-zero model checks the expected
half-period behavior of a circular orbit.

After half an orbital period, a circular two-body trajectory should reach
the position opposite its initial position:

.. math::

   \mathbf{r}\left(\frac{T}{2}\right)
   =
   -\mathbf{r}(0).

The position error is calculated as

.. math::

   e_r =
   \left\|
   \mathbf{r}\left(\frac{T}{2}\right)
   +
   \mathbf{r}(0)
   \right\|.

This provides a basic numerical check of the degree-zero gravity propagation
against the expected circular-orbit geometry.

Trajectory Comparison
---------------------

The difference between the degree-8 and degree-zero position histories is
calculated at each output time:

.. math::

   \Delta r(t)
   =
   \left\|
   \mathbf{r}_{8}(t)
   -
   \mathbf{r}_{0}(t)
   \right\|.

where :math:`\mathbf{r}_{8}` is the degree-8 position and
:math:`\mathbf{r}_{0}` is the point-mass position.

The comparison illustrates the accumulated trajectory separation caused by
the spherical harmonic perturbations included in the degree-8 model.

The plotting routine displays the two propagated trajectories around a
spherical representation of Earth, along with the position difference as a
function of propagation time.

.. figure:: _static/PinesGravity.svg
   :align: center
   :width: 100%

   Comparison of point-mass and degree-8 Pines gravity propagation over
   ten nominal orbital periods.

Earth Surface Visualization
---------------------------

For visualization, a spherical Earth surface is generated using the reference
radius :math:`R_E`. The surface is not used in the gravity calculation; it is
included only to provide a spatial reference for the propagated trajectories.

The surface is constructed from longitude and latitude coordinates:

.. math::

   \begin{aligned}
   x &= R_E\cos(\phi)\cos(\lambda), \\
   y &= R_E\cos(\phi)\sin(\lambda), \\
   z &= R_E\sin(\phi).
   \end{aligned}

where :math:`\phi` is latitude and :math:`\lambda` is longitude.

The resulting surface is plotted with partial transparency so that both
orbital trajectories remain visible.

Full Code
---------

The complete Python implementation is included below.

.. literalinclude:: ../../examples/PinesGravity.py
   :language: python
   :linenos:
   :caption: PinesGravity.py

Implementation Notes
--------------------

This example illustrates several features of ASSET's Pines gravity
implementation:

* **Native gravity evaluation:** The Pines kernel evaluates gravitational
  acceleration using spherical harmonic coefficients from EGM96.
* **Selectable harmonic degree:** The maximum degree can be changed when
  constructing the gravity model.
* **VectorFunction integration:** The gravity model is called directly
  from an ASSET ODE.
* **Derivative evaluation:** The acceleration VectorFunction provides a
  Jacobian through :code:`compute_jacobian`.
* **Trajectory comparison:** The example compares point-mass and
  degree-8 gravity propagation over ten nominal circular-orbit periods.

The example deliberately uses a fixed body-centered frame. A production
high-fidelity Earth gravity propagation would also need to account for the
transformation between the propagation frame and the body-fixed gravity
frame, including Earth rotation and any desired time-dependent Earth
orientation effects.

References
----------

#. Pines, S. "Uniform Representation of the Gravitational Potential and its
   Derivatives." AIAA Journal, 1973.
#. Lemoine, F. G., et al. "The Development of the Joint NASA GSFC and the
   National Imagery and Mapping Agency (NIMA) Geopotential Model EGM96."
   NASA/TP-1998-206861, 1998.
