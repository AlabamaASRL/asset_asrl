The values must be strings or dates or numbers. For the original INPOP/DE files, the floating-points numbers are expressed using a scientific notation in base 10.

 
.. ifconfig:: calcephapi in ('F90', 'F2003')

    Trailing blanks are added to each value.

The following example prints the units of the mission stored in the ephemeris file

.. include:: examples/multiple_getconstantvs.rst
