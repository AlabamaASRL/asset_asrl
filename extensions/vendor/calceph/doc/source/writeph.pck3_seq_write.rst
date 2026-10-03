
This function writes in sequential mode a type 3 segment (Chebyshev polynomials of the angles and their derivatives) in the time scale TDB to the PCK file associated to the ephemeris descriptor *eph*.

The *polynomials* array must be of size record_count*(deg+1)*6 and have the following structure :

.. include:: table_pck3.rst

.. include:: examples/writeph_pck3_seq_write.rst