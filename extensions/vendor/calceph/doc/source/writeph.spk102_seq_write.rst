This function writes in sequential mode a type 102 segment (Chebyshev polynomials of the position only ) in the time scale TCB to the SPK file associated to the ephemeris descriptor *eph*.

The *polynomials* array must be of size record_count*(deg+1)*3 and have the following structure :

.. include:: table_spk2.rst

.. include:: examples/writeph_spk102_seq_write.rst