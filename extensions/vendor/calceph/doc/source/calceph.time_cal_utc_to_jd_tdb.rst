This function converts a Coordinated Universal Time (UTC) calendar date (year, month, day, hour, minute, second) into a Barycentric Dynamical Time (TDB) Julian Date.

The result is returned as two double-precision floating-point numbers (jd0_tdb and jdfrac_tdb) to preserve precision.

This function performs the following transformation chain internally:

1. Converts the UTC calendar date to a UTC Julian Date.

2. Computes the difference between UTC and TT (Terrestrial Time) using leap second data.

3. Converts TT to TDB using |calceph_time_jd_tt_to_jd_tdb|.

This function requires that the ephemeris file |eph| contains both leap second constants (for UTC → TT) and the necessary data for the TT → TDB transformation that can be initialized with |calceph_time_set_relationship_tt_tdb|.

The following example converts a UTC calendar date to TDB:

.. include:: examples/time_cal_utc_to_jd_tdb.rst
