.. _`Time functions`:

.. include:: calceph.time_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_time_jd_to_cal|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function f90calceph_time_jd_to_cal (eph,timescale, jd0,  jdfrac, yy, month, day, hh, min, sec)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param timescale [INTEGER, intent(in)]: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param jd0 [REAL(8), intent(in)]: Integer part of Julian Date
    :param jdfrac [REAL(8), intent(in)]: Fractional part of Julian Date
    :param yy [INTEGER, intent(out)]: Year
    :param month [INTEGER, intent(out)]: Month [1-12]
    :param day [INTEGER, intent(out)]: Day [1-31]
    :param hh [INTEGER, intent(out)]: Hour [0-23]
    :param min [INTEGER, intent(out)]: Minute [0-59]
    :param sec [REAL(8), intent(out)]: Seconds [0-60)
    :r f90calceph_time_jd_to_cal: |retfuncfails0|
    :rtype f90calceph_time_jd_to_cal: INTEGER

.. include:: calceph.time_jd_to_cal.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function f90calceph_time_cal_to_jd(eph, timescale, yy, month, day, hh, min, sec, jd0, jdfrac)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param timescale [INTEGER, intent(in)]: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param yy [INTEGER, intent(in)]: Year
    :param month [INTEGER, intent(in)]: Month [1-12]
    :param day [INTEGER, intent(in)]: Day [1-31]
    :param hh [INTEGER, intent(in)]: Hour [0-23]
    :param min  [INTEGER, intent(in)]: Minute [0-59]
    :param sec [REAL(8), intent(in)]: Seconds [0.0-60.0)
    :param jd0 [REAL(8), intent(out)]: Integer part of Julian Date
    :param jdfrac [REAL(8), intent(out)]: Fractional part of Julian Date
    :r f90calceph_time_cal_to_jd: |retfuncfails0|
    :rtype f90calceph_time_cal_to_jd: INTEGER

.. include:: calceph.time_cal_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_str_to_jd(eph, timescale, const char *str, jd0, jdfrac)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param timescale [INTEGER, intent(in)]: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param str [CHARACTER(len=*), intent(in)]: Input time string to be parsed
    :param jd0 [REAL(8), intent(out)]: Integer part of Julian Date
    :param jdfrac [REAL(8), intent(out)]: Fractional part of Julian Date
    :r f90calceph_time_str_to_jd: |retfuncfails0|
    :rtype f90calceph_time_str_to_jd: INTEGER

.. include:: calceph.time_str_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_set_relationship_tt_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_set_relationship_tt_tdb(eph, model)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param model [INTEGER, intent(in)]: | Relationship model
                                        | 0: use of |calceph_compute_unit|
                                        | 1: use the model based on TLS file
    :r f90calceph_time_set_relationship_tt_tdb: |retfuncfails0|
    :rtype f90calceph_time_set_relationship_tt_tdb: INTEGER

.. include:: calceph.time_set_relationship_tt_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tt|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tdb_to_jd_tt(eph, jd0_tdb, jdfrac_tdb, jd0_tt, jdfrac_tt)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param jd0_tdb [REAL(8), intent(in)]: Integer part of Julian Date (TDB)
    :param jdfrac_tdb [REAL(8), intent(in)]: Fractional part of Julian Date (TDB)
    :param jd0_tt [REAL(8), intent(out)]: Integer part of Julian Date (TT)
    :param jdfrac_tt [REAL(8), intent(out)]: Fractional part of Julian Date (TT)
    :r f90calceph_time_jd_tdb_to_jd_tt: |retfuncfails0|
    :rtype f90calceph_time_jd_tdb_to_jd_tt: INTEGER


.. include:: calceph.time_tdb_to_tt.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tt_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tt_to_jd_tdb(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param jd0_tt [REAL(8), intent(in)]: Integer part of Julian Date (TT)
    :param jdfrac_tt [REAL(8), intent(in)]: Fractional part of Julian Date (TT)
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of Julian Date (TDB)
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of Julian Date (TDB)
    :r f90calceph_time_jd_tt_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_jd_tt_to_jd_tdb: INTEGER

.. include:: calceph.time_tt_to_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tcb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tdb_to_jd_tcb(eph, jd0_tdb, jdfrac_tdb, jd0_tcb, jdfrac_tcb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param jd0_tdb [REAL(8), intent(in)]: Integer part of Julian Date (TDB)
    :param jdfrac_tdb [REAL(8), intent(in)]: Fractional part of Julian Date (TDB)
    :param jd0_tcb [REAL(8), intent(out)]: Integer part of Julian Date (TCB)
    :param jdfrac_tcb [REAL(8), intent(out)]: Fractional part of Julian Date (TCB)
    :r f90calceph_time_jd_tdb_to_jd_tcb: |retfuncfails0|
    :rtype f90calceph_time_jd_tdb_to_jd_tcb: INTEGER


.. include:: calceph.time_tdb_to_tcb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tcb_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tcb_to_jd_tdb(eph, jd0_tcb, jdfrac_tcb, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param jd0_tcb [REAL(8), intent(in)]: Integer part of Julian Date (TCB)
    :param jdfrac_tcb [REAL(8), intent(in)]: Fractional part of Julian Date (TCB)
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of Julian Date (TDB)
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of Julian Date (TDB)
    :r f90calceph_time_jd_tcb_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_jd_tcb_to_jd_tdb: INTEGER

.. include:: calceph.time_tcb_to_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_cal_utc_to_jd_tdb(eph, yy, month, day, hh, min, sec, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param yy [INTEGER, intent(in)]: Year
    :param month [INTEGER, intent(in)]: Month
    :param day [INTEGER, intent(in)]: Day
    :param hh [INTEGER, intent(in)]: Hour
    :param min  [INTEGER, intent(in)]: Minute
    :param sec [REAL(8), intent(in)]: Second
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of the resulting TDB Julian Date
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of the resulting TDB Julian Date
    :r f90calceph_time_cal_utc_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_cal_utc_to_jd_tdb: INTEGER

.. include:: calceph.time_cal_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_cal_utc|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tdb_to_cal_utc(eph, jd0_tdb, jdfrac_tdb, yy, month, day, hh, min, sec)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param jd0_tdb [REAL(8), intent(in)]: Integer part of the TDB Julian Date
    :param jdfrac_tdb [REAL(8), intent(in)]: Fractional part of the TDB Julian Date
    :param yy [INTEGER, intent(out)]: Year
    :param month [INTEGER, intent(out)]: Month
    :param day [INTEGER, intent(out)]: Day
    :param hh [INTEGER, intent(out)]: Hour
    :param min  [INTEGER, intent(out)]: Minute
    :param sec [REAL(8), intent(out)]: Second
    :r f90calceph_time_jd_tdb_to_cal_utc: |retfuncfails0|
    :rtype f90calceph_time_jd_tdb_to_cal_utc: INTEGER

.. include:: calceph.time_jd_tdb_to_cal_utc.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_str_utc_to_jd_tdb(eph, const char *str, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param str [CHARACTER(len=*), intent(in)]: UTC time string to parse
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of the resulting TDB Julian Date
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of the resulting TDB Julian Date
    :r f90calceph_time_str_utc_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_str_utc_to_jd_tdb: INTEGER

.. include:: calceph.time_str_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_any_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_str_any_to_jd_tdb(eph, const char *str, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param str [CHARACTER(len=*), intent(in)]: Time string to parse
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of the resulting TDB Julian Date
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of the resulting TDB Julian Date
    :r f90calceph_time_str_any_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_str_any_to_jd_tdb: INTEGER

.. include:: calceph.time_str_any_to_jd_tdb.rst


.. %----------------------------------------------------------------------------

|menu_calceph_time_str_spacecraft_clock_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


 .. f:function:: function f90calceph_time_str_spacecraft_clock_to_jd_tdb( eph, itarget, str, jd0_tdb, jdfrac_tdb)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param itarget [INTEGER, intent(in)]: NAIF ID of the spacecraft
    :param str [CHARACTER(len=*), intent(in)]:  Spacecraft clock string (e.g., "1/1234:56")
    :param jd0_tdb [REAL(8), intent(out)]: Integer part of the resulting TDB Julian Date
    :param jdfrac_tdb [REAL(8), intent(out)]: Fractional part of the resulting TDB Julian Date
    :r f90calceph_time_str_spacecraft_clock_to_jd_tdb: |retfuncfails0|
    :rtype f90calceph_time_str_spacecraft_clock_to_jd_tdb: INTEGER

.. include:: calceph.time_str_sclk_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_str_spacecraft_clock|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 .. f:function:: function f90calceph_time_jd_tdb_to_str_spacecraft_clock(eph, itarget,  jd0_tdb,  jdfrac_tdb, str)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param itarget [INTEGER, intent(in)]: NAIF ID of the spacecraft
    :param jd0_tdb [REAL(8), intent(in)]:: Integer part of Julian Date (TDB)
    :param jdfrac_tdb[REAL(8), intent(in)]: : Fractional part of Julian Date (TDB)
    :param  str [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), intent(out)]: Output spacecraft clock time string
    :r f90calceph_time_jd_tdb_to_str_spacecraft_clock: |retfuncfails0|
    :rtype f90calceph_time_jd_tdb_to_str_spacecraft_clock: INTEGER

.. include:: calceph.time_jd_tdb_to_str_sclk.rst
