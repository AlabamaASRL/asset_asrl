This function is similar to the function |calceph_orient_unit|, except that the order of the computed derivatives is specified. 


This function reads, if needed, in the ephemeris file |eph| and interpolates the angular momentum vector due to the rotation of the body, divided by the product of the mass :math:`m` and of the square of the radius :math:`R`, of a single body (*target*) for the time *JD0+time* and stores the results to *PVAJ*. The angular momentum :math:`L` , due to the rotation of the body, is defined as the product of the inertia matrix :math:`I` by the angular velocity vector :math:`{\omega}`. So the returned value is :math:`L/(mR^2)=(I\omega)/(mR^2)`
The order of the derivatives are specified by *order*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|. 
The output values are expressed in the units specified by *unit*.

.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.

This function checks the units if invalid combinations of units are given to the function.

.. ifconfig:: calcephapi in ('C')

    The returned array *PVAJ*  must be large enough to store the results. The size of this array must be equal to 3*(order+1).

        - PVAJ[0..2] contain the angular momentum  and is always valid.
        - PVAJ[3..5] contain the first derivative and is only valid if *order* is greater or equal to 1.
        - PVAJ[6..8] contain the second derivative and is only valid if *order* is greater or equal to 2.
        - PVAJ[9..11] contain the third derivative and is only valid if *order* is equal to 3.

.. ifconfig:: calcephapi in ('F90','F2003','Python','Mex')

    The returned array *PVAJ* must be large enough to store the results.

        -  PVAJ[1:3] contain the angular momentum  and is always valid.
        -  PVAJ[4:6] contain the first derivative and is only valid if *order* is greater or equal to 1.
        -  PVAJ[7:9] contain the second derivative and is only valid if *order* is greater or equal to 2.
        -  PVAJ[10:12] contain the third derivative and is only valid if *order* is equal to 3.

The values stored in the  array *PVAJ* are expressed in the following units

    - The angular momentum and its derivatives are expressed in days if unit contains |CALCEPH_UNIT_DAY|.
    - The angular momentum and its derivatives are expressed in seconds if unit contains |CALCEPH_UNIT_SEC|.

|timescale_JD0_Time|


The following example prints only the angular momentum, due to its rotation, of the Earth at time=2451419.5

.. include:: examples/multiple_rotangmom_order.rst
    