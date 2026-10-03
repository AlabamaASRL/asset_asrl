This function writes in sequential mode a type 103 segment (Chebyshev polynomials of the position and velocity ) in the time scale TCB to the SPK file associated to the ephemeris descriptor *eph*.


The *polynomials* array must be of size record_count*(deg+1)*6 and have the following structure :

.. include:: table_spk3.rst

.. include:: examples/writeph_spk103_seq_write.rst