.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.


This function is similar to the function |calceph_compute_unit|, except that the order of the computed derivatives is specified. 

This function reads, if needed, in the ephemeris file |eph| and interpolates a single object, usually the position and their derivatives of one body (*target*) relative to another (*center*) for the time *JD0+time* and stores the results to *PVAJ*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|.
The order of the derivatives are specified by *order*. The output values are expressed in the units specified by *unit*.

The returned array *PVAJ* has the following properties

  - If the target is the time scale transformation TT-TDB, only the first elements of each component will get the result. 
  - If the target is the time scale transformation  *TCG-TCB*, only the first elements of each component will get the result. 
  - If the target is *Librations*, the array contains the angles of the librations of the Moon  and their successive derivatives.
  - If the target is *Nutations*, the array contains the nutation angles and their successive derivatives.
  - Otherwise the returned value is the cartesian position (x,y,z), the velocity (xdot, ydot, zdot), the jerk and the acceleration. 

.. ifconfig:: calcephapi in ('C')

    The returned array *PVAJ*  must be large enough to store the results. The size of this array must be equal to 3*(order+1).

        - PVAJ[0..2] contain the position (x,y,z) and is always valid.
        - PVAJ[3..5] contain the velocity  (dx/dt,dy/dt,dz/dt) and is only valid if *order* is greater or equal to 1.
        - PVAJ[6..8] contain the acceleration (d^2x/dt^2,d^2y/dt^2,d^2z/dt^2) and is only valid if *order* is greater or equal to 2.
        - PVAJ[9..11] contain the jerk (d^3x/dt^3,d^3y/dt^3,d^3z/dt^3) and is only valid if *order* is equal to 3.


.. ifconfig:: calcephapi in ('F90','F2003','Python','Mex')

    The returned array *PVAJ* must be large enough to store the results.

        - PVAJ[1:3] contain the position (x,y,z) and is always valid.
        - PVAJ[4:6] contain the velocity  (dx/dt,dy/dt,dz/dt) and is only valid if *order* is greater or equal to 1.
        - PVAJ[7:9] contain the acceleration (d^2x/dt^2,d^2y/dt^2,d^2z/dt^2) and is only valid if *order* is greater or equal to 2.
        - PVAJ[10:12] contain the jerk (d^3x/dt^3,d^3y/dt^3,d^3z/dt^3) and is only valid if *order* is equal to 3.

.. |times|  unicode:: U+000D7 .. MULTIPLICATION SIGN
.. ifconfig:: calcephapi in ('Python')

   If *JD0* and *time* are list or NumPy's array (1D) of double-precision floating-point values, the returned array *PVAJ* is a list of 3*(order+1) arrays. Each array contain a single component of position, velocity ... (e.g., PV[0] contains the coordinate X, PV[1] contains the coordinate Y, ...) . 

|timescale_JD0_Time|

.. warning::
    |warning_UTC|

The values stored in the  array *PVAJ* are expressed in the following units

  - The position, velocity, acceleration and jerk are expressed in Astronomical Unit (au) if unit contains |CALCEPH_UNIT_AU|.
  - The position, velocity, acceleration and jerk are expressed in kilometers if unit contains |CALCEPH_UNIT_KM|.
  - The velocity, acceleration, jerk, TT-TDB, TCG-TCB or the derivatives of the angles of the librations of the Moon are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
  - The velocity, acceleration, jerk, TT-TDB, TCG-TCB or the derivatives of the angles of the librations of the Moon are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.
  - The angles of the librations of the Moon are expressed in radians if unit contains |CALCEPH_UNIT_RAD|.

For example, to get the positions, velocities, accelerations and jerks expressed in kilometers and kilometers/seconds, the unit must be set to |CALCEPH_UNIT_KM| + |CALCEPH_UNIT_SEC|. 


This function checks the units if invalid combinations of units are given to the function.



The following example prints the heliocentric coordinates of Mars at time=2442457.5

.. include:: examples/multiple_compute_order.rst
