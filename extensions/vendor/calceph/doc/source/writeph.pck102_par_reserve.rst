
This function is meant to enable the parallel writing (with multiple threads) of several type 102 segments (Chebyshev polynomials of the angle only ) in the time scale TCB to the PCK file associated to the ephemeris descriptor *eph*.

It reserves space in the file for *target_count* segments, each segment associated to a target in the *targets* array.
Each target has its own interpolation interval length specified in the *intlens_jd_tdb* array, its own number of records specified in the *record_counts* array, and its own segment identifier specified in the *segids* array.
All the targets share the same *frame*, *start_..._tdb*, *end_..._tdb* and *deg* parameters.

The arrays *targets*, *intlens_jd_tdb*, *record_counts* and *segids* must be of size *target_count*.

The reservation *id* returned by this function must be used in the function |writeph_pck102_par_write| to write in the corresponding reserved space.
In addition, the function |writeph_pck102_par_write| will require a *target_index*, which is the index of the target in the *targets* array used in this function.

A call to |writeph_pck102_par_reserve| is always performed by a single thread and is followed by several calls to |writeph_pck102_par_write| by one or several threads.


.. warning::

    The data covers the same timespan from the date *start_jd_tcb+start_frac_tcb* to *end_jd_tcb+end_frac_tcb* for all targets, but each target may have a different number of polynomials.
    The time span *record_counts\*intlens_jd_tcb* must be equal to the timespan defined from the date *start_jd_tcb+start_frac_tcb* to *end_jd_tcb+end_frac_tcb*.
    If the targets doesnot have the timespan, multiple reservations must be performed before the call to writing |writeph_pck102_par_write|. 


.. include:: examples/writeph_pck102_par_reserve.rst