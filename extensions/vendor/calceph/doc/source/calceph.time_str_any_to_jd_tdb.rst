This function parses a time string, identifies its native time scale, and converts the date directly to the TDB scale.

The function first parses the string to extract the date and the time scale (e.g., "2000-01-01 TAI"). Based on the detected time scale, it performs the necessary conversions to reach TDB.

The supported input timescales and their internal conversion paths are:

- |CALCEPH_TDB|: No conversion needed.
- |CALCEPH_TT|: Performs the transformation TT → TDB.
- |CALCEPH_TAI|: Performs the transformation chain TAI → TT → TDB.
- |CALCEPH_UTC|: Performs the transformation chain UTC → TAI → TT → TDB.

If the input string does not specify a time scale, the function defaults to |CALCEPH_UTC| before converting to TDB.

This function requires that the ephemeris file |eph| contains the necessary data for the requested conversions (leap seconds for UTC/TAI, and relationship model for TT/TDB that can be initialized with |calceph_time_set_relationship_tt_tdb|).

The following example converts a TAI string to TDB:

.. include:: examples/time_str_any_to_jd_tdb.rst
