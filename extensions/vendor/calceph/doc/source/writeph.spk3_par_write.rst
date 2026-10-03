This function writes in parallel mode records of a type 3 segment (Chebyshev polynomials of the position and velocity) in the time scale TDB to the SPK file associated to the ephemeris descriptor *eph*.

The *reservation* parameter must be the reservation id returned by the function |writeph_spk3_par_reserve|.

The *target_index* parameter is the index of the target in the *targets* array that was passed to the function |writeph_spk3_par_reserve|, starting from 0.

The *record_begin_index* parameter is the index of the first record to be written in the target's segment, starting from 0.

The *record_count* parameter is the number of records to be written starting from the *record_begin_index*.

The *polynomials* array must be of size record_count*(deg+1)*6 and have the following structure :

.. include:: table_spk3.rst

With the 1st record being the record at index *record_begin_index*, and the (record_count)th record being the record at index *record_begin_index* + *record_count* - 1.

Multiple threads can call this function at the same time, but with different values of *target_index* and/or *record_begin_index*. The behavior is undefined if multiple threads overlap the written data : e.g., two threads write to the same target *target_index* and a common record number.
