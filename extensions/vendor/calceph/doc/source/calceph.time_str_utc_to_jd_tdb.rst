This function parses a time string representing a UTC date and converts it directly to a TDB Julian Date.

The function forces the interpretation of the input string as UTC, ignoring any potential timescale suffix present in the string. It then performs the full conversion chain (UTC → TAI → TT → TDB).

The following example converts a UTC string to TDB:

.. include:: examples/time_str_utc_to_jd_tdb.rst
