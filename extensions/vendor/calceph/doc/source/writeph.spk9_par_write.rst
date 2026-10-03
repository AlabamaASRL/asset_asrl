This function writes in parallel mode records of a type 9 segment (discrete states of the positions and velocities, Lagrange interpolation with unequal timesteps) to the SPK file associated to the ephemeris descriptor *eph*.


The *reservation* parameter must be the reservation id returned by the function |writeph_spk9_par_reserve|.

The *target_index* parameter is the index of the target in the *targets* array that was passed to the function |writeph_spk9_par_reserve|, starting from 0.

The *record_begin_index* parameter is the index of the first record to be written in the target's segment, starting from 0.

The *record_count* parameter is the number of records to be written starting from the *record_begin_index*.

The *states* array must be of size record_count*6 and have the following structure :

.. include:: table_spk8.rst
    
The *epochs* array must be of size record_count and contain the epochs (in Julian date TDB) corresponding to each state vector, it has the following structure :

.. include:: table_spk9.rst


Multiple threads can call this function at the same time, but with different values of *target_index* and/or *record_begin_index*. The behavior is undefined if multiple threads overlap the written data : e.g., two threads write to the same target *target_index* and a common record number.
