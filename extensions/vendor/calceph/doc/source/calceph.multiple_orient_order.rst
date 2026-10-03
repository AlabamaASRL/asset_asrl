.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.


This function is similar to the function |calceph_orient_unit|, except that the order of the computed derivatives is specified. 


This function reads, if needed, in the ephemeris file |eph| and interpolates the orientation of a single body (*target*) for the time *JD0+time* and stores the results to *PVAJ*. 
The order of the derivatives are specified by *order*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|. 
The output values are expressed in the units specified by *unit*.

This function checks the units if invalid combinations of units are given to the function.

The returned array *PVAJ* has the following properties

 - If *unit* contains |CALCEPH_OUTPUT_NUTATIONANGLES|, the array contains the nutation angles and their successive derivatives for the orientation of the body. At the present moment, only the nutation for the earth are supported in the original DE files.
 - If *unit* contains |CALCEPH_OUTPUT_EULERANGLES|, or doesnot contain |CALCEPH_OUTPUT_NUTATIONANGLES|, the array contains the euler angles and their successive derivatives for the orientation of the body. 

.. ifconfig:: calcephapi in ('C')

    The returned array *PVAJ*  must be large enough to store the results. The size of this array must be equal to 3*(order+1).

        - PVAJ[0..2] contain the angles  and is always valid.
        - PVAJ[3..5] contain the first derivative and is only valid if *order* is greater or equal to 1.
        - PVAJ[6..8] contain the second derivative and is only valid if *order* is greater or equal to 2.
        - PVAJ[9..11] contain the third derivative and is only valid if *order* is equal to 3.

.. ifconfig:: calcephapi in ('F90','F2003','Python','Mex')

    The returned array *PVAJ* must be large enough to store the results.

        -  PVAJ[1:3] contain the angles  and is always valid.
        -  PVAJ[4:6] contain the first derivative and is only valid if *order* is greater or equal to 1.
        -  PVAJ[7:9] contain the second derivative and is only valid if *order* is greater or equal to 2.
        -  PVAJ[10:12]  contain the third derivative and is only valid if *order* is equal to 3.

.. ifconfig:: calcephapi in ('Python')

   If *JD0* and *time* are list or NumPy's array (1D) of double-precision floating-point values, the returned array *PVAJ* is a list of 3*(order+1) arrays. Each array contain a single component of the orientation. 


    
The values stored in the  array *PVAJ* are expressed in the following units

  - The derivatives of the angles are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
  - The derivatives of the angles are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.
  - The angles and their derivatives are expressed in radians if unit contains |CALCEPH_UNIT_RAD|.

|timescale_JD0_Time|


The following example prints only the angles of libration of the Moon at time=2442457.5

.. include:: examples/multiple_orient_order.rst
    