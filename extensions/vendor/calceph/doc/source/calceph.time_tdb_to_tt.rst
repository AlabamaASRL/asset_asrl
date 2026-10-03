This function converts a Julian Date from the Barycentric Dynamical Time (TDB) scale to the Terrestrial Time (TT) scale.

The conversion relies on the model previously set by the function |calceph_time_set_relationship_tt_tdb|.

The input Julian Date is provided as two double-precision numbers (jd0_tdb and jdfrac_tdb) to maintain precision. The result is returned in the output variables jd0_tt and jdfrac_tt.

Depending on the selected model:

- Model 0: Retrieves the offset with |calceph_compute_unit|.

- Model 1: Uses the internal default model from the TLS file.

The following example converts a TDB date to TT:

.. include:: examples/time_jd_tdb_to_jd_tt.rst
