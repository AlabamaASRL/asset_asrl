
This function reads, if needed, in the ephemeris file |eph| and interpolates the angular momentum vector due to the rotation of the body, divided by the product of the mass :math:`m` and of the square of the radius :math:`R`, of a single body (*target*) for the time *JD0+time* and stores the results to *PV*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|. The angular momentum :math:`L` , due to the rotation of the body, is defined as the product of the inertia matrix :math:`I` by the angular velocity vector :math:`{\omega}`. So the returned value is :math:`L/(mR^2)=(I\omega)/(mR^2)`

|timescale_JD0_Time|

The output values are expressed in the units specified by *unit*.

.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.

This function checks the units if invalid combinations of units are given to the function.

The values stored in the  array *PV* are expressed in the following units

    - The angular momentum and its derivative are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
    - The angular momentum and its derivative are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.


The following example prints the angular momentum, due to its rotation, for the Earth at time=2451419.5

.. include:: examples/multiple_rotangmom_unit.rst