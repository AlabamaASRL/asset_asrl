This function is meant to enable the parallel writing (with multiple threads) of several type 13 segments (discrete states of the positions and velocities, Hermite interpolation with variable timesteps) to the SPK file associated to the ephemeris descriptor *eph*.

It reserves space in the file for *target_count* segments, each segment associated to a target in the *targets* array.
Each target has its own number of records specified in the *record_counts* array, and its own segment identifier specified in the *segids* array.
All the targets share the same *center*, *frame*, *start_..._tdb*, *end_..._tdb* and *deg* parameters.

The array *targets*, *record_counts* and *segids* must be of size *target_count*.

The reservation *id* returned by this function must be used in the function |writeph_spk13_par_write| to write in the corresponding reserved space.
In addition, the function |writeph_spk13_par_write| will require a *target_index*, which is the index of the target in the *targets* array used in this function.

A call to *writeph_spk13_par_reserve* is always performed by a single thread and is followed by several calls to |writeph_spk13_par_write| by one or several threads.

.. include:: examples/writeph_spk13_par_reserve.rst