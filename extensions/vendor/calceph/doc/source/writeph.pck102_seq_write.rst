
This function writes in sequential mode a type 102 segment (Chebyshev polynomials of the angle only) in the time scale TCB to the PCK file associated to the ephemeris descriptor *eph*.

The *polynomials* array must be of size record_count*(deg+1)*3 and have the following structure :

.. include:: table_pck2.rst

.. include:: examples/writeph_pck102_seq_write.rst