
This function returns the value associated to the constant *name* in the header of the ephemeris file |eph|. Only the first value is returned if multiple values are associated to a constant, such as a list of values.


This function is the same function as |calceph_getconstantsd|.

The following example prints the value of the astronomical unit stored in the ephemeris file

.. include:: examples/multiple_getconstant.rst