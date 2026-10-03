This function converts a Julian Date, expressed as two double-precision numbers, into a Gregorian calendar date (year, month, day) and time (hour, minute, second).

Depending on the value of the timescale argument, the function behaves as follows:

- |CALCEPH_UTC|: The function handles leap seconds. It requires that the ephemeris file |eph| contains leap second history (constants `DELTET` or `DELTA_AT`).

- Other timescales (e.g., |CALCEPH_TT|, |CALCEPH_TDB|): The time is treated as continuous. The seconds field is always in the range [0, 60[.

If the timescale is |CALCEPH_UTC| and the ephemeris file does not contain the required leap second constants, the function returns an error.

The following example converts a JD to a calendar date in UTC:

.. include:: examples/time_jd_to_cal.rst
