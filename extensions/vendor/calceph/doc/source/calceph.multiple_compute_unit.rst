.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.

This function is similar to the function |calceph_compute|, except that the units of the output are specified. 

This function reads, if needed, in the ephemeris file |eph| and interpolates a single object, usually the position and velocity of one body (*target*) relative to another (*center*) for the time *JD0+time* and stores the results to *PV*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|.
The output values are expressed in the units specified by *unit*.

This function checks the units if invalid combinations of units are given to the function.


|timescale_JD0_Time|

.. warning::
    |warning_UTC|

The returned array *PV* has the following properties

  - If the target is the time scale transformation TT-TDB, only the first element of this array will get the result. 
  - If the target is the time scale transformation  *TCG-TCB*, only the first element of this array will get the result. 
  - If the target is *Librations*, the array contains the angles of the librations of the Moon  and their derivatives.
  - If the target is *Nutations*, the array contains the nutation angles and their derivatives.
  - Otherwise the returned value is the cartesian position (x,y,z) and the velocity (xdot, ydot, zdot). 

.. ifconfig:: calcephapi in ('Python')

   If *JD0* and *time* are list or NumPy's array (1D) of double-precision floating-point values, the returned array *PV* is a list of 6 arrays. Each array contain a single component of position or velocity (e.g., PV[0] contains the coordinate X, PV[1] contains the coordinate Y, ...) . 


The values stored in the  array *PV* are expressed in the following units

 - The position and velocity are expressed in Astronomical Unit (au) if unit contains |CALCEPH_UNIT_AU|.
 - The position and velocity are expressed in kilometers if unit contains |CALCEPH_UNIT_KM|.
 - The velocity, TT-TDB, TCG-TCB, the derivatives of the angles of the nutation, or the derivatives of the librations of the Moon or are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
 - The velocity, TT-TDB, TCG-TCB, the derivatives of the angles of the nutation, or the derivatives of the librations of the Moon are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.
 - The angles of the librations of the Moon or the nutation angles are expressed in radians if unit contains |CALCEPH_UNIT_RAD|.

For example, to get the position and velocities expressed in kilometers and kilometers/seconds, the unit must be set to |CALCEPH_UNIT_KM| + |CALCEPH_UNIT_SEC|. 



The following example prints the heliocentric coordinates of Mars  at time=2442457.5

.. include:: examples/multiple_compute_unit.rst