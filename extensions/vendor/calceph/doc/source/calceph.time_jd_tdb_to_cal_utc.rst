This function converts a TDB Julian Date into a UTC calendar date (year, month, day, hour, minute, second).

This is the inverse operation of |calceph_time_cal_utc_to_jd_tdb|.

The transformation chain performed is:

1. Converts TDB to TT using |calceph_time_jd_tdb_to_jd_tt|.

2. Computes the difference between TT and UTC (using leap seconds).

3. Converts the resulting UTC Julian Date into calendar components.

The function requires that the ephemeris file |eph| contains the necessary time constants.

The following example converts a TDB Julian Date to a UTC calendar date:

.. include:: examples/time_jd_tdb_to_cal_utc.rst
