This function writes in sequential mode a type 12 segment (discrete states of the positions and velocities, Hermite interpolation with equal timesteps) to the SPK file associated to the ephemeris descriptor *eph*.

The *states* array must be of size record_count*6 and have the following structure :

.. include:: table_spk8.rst

.. include:: examples/writeph_spk12_seq_write.rst