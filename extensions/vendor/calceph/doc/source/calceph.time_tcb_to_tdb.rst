This function converts a Julian Date from the Barycentric Coordinate Timescale (TCB)  to the Barycentric Dynamical Timescale (TDB) .

This is the inverse operation of |calceph_time_jd_tdb_to_jd_tcb|. 

The conversion relies on the relation between TCB and TDB defined by the IAU 2006 Resolution B3 : "Re-definition of Barycentric Dynamical Time, TDB".


The input Julian Date is provided as two double-precision numbers (jd0_tcb and jdfrac_tcb). The converted date is stored in jd0_tdb and jdfrac_tdb.

If the ephemeris handle is NULL, the function returns an error.

The following example converts a TCB date to TDB:

.. include:: examples/time_jd_tcb_to_jd_tdb.rst
