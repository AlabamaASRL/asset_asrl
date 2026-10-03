This function writes in sequential mode a type 13 segment (discrete states of the positions and velocities, Hermite interpolation with unequal timesteps) to the SPK file associated to the ephemeris descriptor *eph*.

The *states* array must be of size record_count*6 and have the following structure :

.. include:: table_spk8.rst
    
The *epochs* array must be of size record_count and contain the epochs (in Julian date TDB) corresponding to each state vector, it has the following structure :

.. include:: table_spk9.rst

.. include:: examples/writeph_spk13_seq_write.rst