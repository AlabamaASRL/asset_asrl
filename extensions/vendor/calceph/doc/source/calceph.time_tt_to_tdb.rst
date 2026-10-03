This function converts a Julian Date from the Terrestrial Time (TT) scale to the Barycentric Dynamical Time (TDB) scale.

This is the inverse operation of |calceph_time_jd_tdb_to_jd_tt|. It uses the model configured with the function |calceph_time_set_relationship_tt_tdb|.

The input Julian Date is provided as two double-precision numbers (jd0_tt and jdfrac_tt). The converted date is stored in jd0_tdb and jdfrac_tdb.

If the ephemeris handle is NULL or if the model configuration is invalid, the function returns an error.

The following example converts a TT date to TDB:

.. include:: examples/time_jd_tt_to_jd_tdb.rst
