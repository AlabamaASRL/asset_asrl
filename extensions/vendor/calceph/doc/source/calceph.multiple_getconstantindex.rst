This function returns the name and its value of the constant available at the specified index in the header of the ephemeris file |eph|. The value of *index* must be between 1 and |calceph_getconstantcount|.

Only the first value is returned if multiple values are associated to a constant, such as a list of values. If the first value is not an floating-point number, such as a string, then the function returns 0 without raising an error. 

.. ifconfig:: calcephapi in ('F90')

    Trailing blanks are added to the name of the constant.

The following example displays the name of the constants, stored in the ephemeris file, and their values 

.. include:: examples/multiple_getconstantindex.rst