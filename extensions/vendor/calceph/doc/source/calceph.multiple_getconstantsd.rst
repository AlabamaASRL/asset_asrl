This function returns, as a floating-point number, the value associated to the constant *name* in the header of the ephemeris file |eph|. Only the first value is returned if multiple values are associated to a constant, such as a list of values. The value must be a floating-point or integer number, otherwise an error is reported. 

This function is the same function as |calceph_getconstant|.

The following example prints the value of the astronomical unit stored in the ephemeris file

.. include:: examples/multiple_getconstantsd.rst