This function parses a time string and converts it into a Julian Date, returned as two double-precision numbers (jd0 and jdfrac).

This function supports various input string formats, including ISO 8601, explicit Julian Dates (prefixed with "JD"), and standard calendar formats.

The interpretation of the date depends on the timescale argument:

- Specific Timescale (e.g., |CALCEPH_UTC|, |CALCEPH_TT|): The function interprets the date components parsed from the string as being in this specific timescale, regardless of any timescale suffix present in the string itself.

- |CALCEPH_TIMESCALE_FROM_STR|: The function attempts to detect the timescale from the string (e.g., "2000-01-01 TDB").

    - If a timescale is found in the string, it is used for the conversion.

    - If no timescale is found, the function defaults to UTC.

If the resulting conversion requires UTC (either explicitly requested or detected), the ephemeris handle |eph| must contain leap second data.

The following example parses a string and converts it to a Julian Date:

.. include:: examples/time_str_to_jd.rst
