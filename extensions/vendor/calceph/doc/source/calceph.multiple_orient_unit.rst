This function reads, if needed, in the ephemeris file |eph| and interpolates the orientation of a single body (*target*) for the time *JD0+time* and stores the results to *PV*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|.
The output values are expressed in the units specified by *unit*.

.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.

|timescale_JD0_Time|

This function checks the units if invalid combinations of units are given to the function.

The returned array *PV* has the following properties

 - If *unit* contains |CALCEPH_OUTPUT_NUTATIONANGLES|,  the array contains the nutation angles and their derivatives for the orientation of the body. At the present moment, only the nutation for the earth are supported in the original DE files.
 - If *unit* contains |CALCEPH_OUTPUT_EULERANGLES|, or doesnot contain |CALCEPH_OUTPUT_NUTATIONANGLES|, the array contains the euler angles and their derivatives for the orientation of the body. 

.. ifconfig:: calcephapi in ('Python')

   If *JD0* and *time* are list or NumPy's array (1D) of double-precision floating-point values, the returned array *PV* is a list of 6 arrays. Each array contain a single component of orientation. 

The values stored in the  array *PV* are expressed in the following units

  - The derivatives of the angles are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
  - The derivatives of the angles are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.
  - The angles and their derivatives are expressed in radians if unit contains |CALCEPH_UNIT_RAD|.

For example, to get the nutation angles of the Earth and their derivatives expressed in radian and radian/seconds using the NAIF identification numbering system, the target must be set to NAIFID_EARTH and the unit must be set to |CALCEPH_OUTPUT_NUTATIONANGLES| + |CALCEPH_UNIT_RAD| + |CALCEPH_UNIT_SEC|. 

The following example prints the angles of libration of the Moon at time=2442457.5

.. include:: examples/multiple_orient_unit.rst
    