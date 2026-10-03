This function writes in sequential mode a type 8 segment (discrete states of the positions and velocities, Lagrange interpolation with equal timesteps) in the time scale TDB to the SPK file associated to the ephemeris descriptor *eph*.

The *states* array must be of size record_count*6 and have the following structure :

.. include:: table_spk8.rst

.. include:: examples/writeph_spk8_seq_write.rst