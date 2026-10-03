This function converts a Julian Date from the Barycentric Dynamical Timescale (TDB)  to the  Barycentric Coordinate Timescale (TCB).

The conversion relies on the relation between TCB and TDB defined by the IAU 2006 Resolution B3 : "Re-definition of Barycentric Dynamical Time, TDB".

The input Julian Date is provided as two double-precision numbers (jd0_tdb and jdfrac_tdb) to maintain precision. The result is returned in the output variables jd0_tcb and jdfrac_tcb.


The following example converts a TDB date to TCB:

.. include:: examples/time_jd_tdb_to_jd_tcb.rst
