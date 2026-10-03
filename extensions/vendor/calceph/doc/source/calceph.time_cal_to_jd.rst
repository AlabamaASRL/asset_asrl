This function converts a Gregorian calendar date (year, month, day) and time (hour, minute, second) into a Julian Date, expressed as two double-precision floating-point numbers. 

The resulting Julian Date is the sum of the two returned values jd0 and jdfrac. This split representation preserves numerical precision.

Depending on the value of the timescale argument, the function performs the conversion as follows:

- |CALCEPH_UTC|: The function accounts for leap seconds. It calculates the Julian Date by checking the leap second history stored in the ephemeris file (eph).

    - If the input date corresponds to a leap second (e.g., 23:59:60), the function correctly computes the JD for that instant.

    - The function requires the ephemeris file to contain standard time constants (`DELTET` or `DELTA_AT`).

- Other timescales (e.g., |CALCEPH_TT|, |CALCEPH_TDB|): The time is treated as continuous (without leap seconds). The minutes are assumed to always contain 60 seconds.

If timescale is |CALCEPH_UTC| and the necessary leap second data is missing from the ephemeris file, the function returns an error.

The following example converts a calendar date to a Julian Date:

.. include:: examples/time_cal_to_jd.rst
