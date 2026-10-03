This function writes in sequential mode a type 2 segment (Chebyshev polynomials of Chebyshev polynomials of the angles only) in the time scale TDB to the PCK file associated to the ephemeris descriptor *eph*.

The *polynomials* array must be of size record_count*(deg+1)*3 and have the following structure :

.. include:: table_pck2.rst

.. include:: examples/writeph_pck2_seq_write.rst