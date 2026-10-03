
Time conversion functions
=========================

The following group of functions allow to convert dates between different timescales (UTC, TAI, TT, TDB, TCB) and formats (Julian date, calendar date).
Some of these functions accept time strings as input, which are parsed internally.

When an error occurs, these functions execute error handlers according to the behavior defined by the function |calceph_seterrorhandler|.

.. %------------------------------------------------------------------------

Timescale notes
---------------

Conversions involving the Coordinated Universal Time (UTC) rely on the history of leap seconds.
Consequently, these functions require an ephemeris file containing the time constants (usually DELTET or DELTA_AT) to be opened and provided to the function . These time constants are provided in the leap second spice kernel file (\*.tls).

Conversions the Barycentric Dynamical Time (TDB) and the Terrestrial Time (TT) are performed using the model chosen with the |calceph_time_set_relationship_tt_tdb| function.

Input string formats
--------------------

For functions accepting time strings as input, the library supports various formats, including ISO 8601, explicit Julian dates, and custom calendar strings.

If the time scale is specified within the string (e.g., "1996-01-01 UTC"), it is automatically detected.
If the time scale is not present in the string, it is considered as UTC time scale.

The following strings show some examples of supported strings

::

    "2006-01-15T12:01:10.00 UTC"
    "2006-01-15T12:01:10.00 TDB"
    "1995 December 31  15:59:59.5  (UTC)"
    "1988 June 13, 12:29:48 TDB"
    "1988 June 13, 12:29:48 TT"
    "JD 2453751.0008101849816740 (TAI)"
    "JD 2453751.0011826851405203 (TT)"
    "JD 2453751.0013471459969878 (TCB)"
    "2451515.2981 (JD)"
