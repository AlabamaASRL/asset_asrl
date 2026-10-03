.. _`Time functions`:

.. include:: calceph.time_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_time_jd_to_cal|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_to_cal (t_calcephbin * eph, int timescale, double jd0, double jdfrac, int *yy, int *month, int *day, int *hh, int *min, double *sec)

    :param  eph: |arg_eph|
    :param timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param jd0: Integer part of Julian Date
    :param jdfrac: Fractional part of Julian Date
    :param yy: Year
    :param month: Month [1–12]
    :param day: Day [1–31]
    :param hh: Hour [0–23]
    :param min: Minute [0–59]
    :param sec: seconds [0.0–61.0[
    :return: |retfuncfails0|

.. include:: calceph.time_jd_to_cal.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_cal_to_jd(t_calcephbin * eph, int timescale, int yy, int month, int day, int hh, int min, double sec, double *jd0, double *jdfrac)

    :param  eph: |arg_eph|
    :param timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param yy: Year
    :param month: Month [1–12]
    :param day: Day [1–31]
    :param hh: Hour [0–23]
    :param min: Minute [0–59]
    :param sec: Seconds [0.0–61.0[
    :param jd0: Integer part of Julian Date
    :param jdfrac: Fractional part of Julian Date
    :return: |retfuncfails0|

.. include:: calceph.time_cal_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_str_to_jd(t_calcephbin * eph, int timescale, const char *str, double *jd0, double *jdfrac)

    :param  eph: |arg_eph|
    :param timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param str: Input time string to be parsed
    :param jd0: Integer part of Julian Date
    :param jdfrac: Fractional part of Julian Date
    :return: |retfuncfails0|

.. include:: calceph.time_str_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_set_relationship_tt_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_set_relationship_tt_tdb(t_calcephbin * eph, int model)

    :param  eph: |arg_eph|
    :param model: | Relationship model
                  | 0: use of |calceph_compute_unit|
                  | 1: use the model based on TLS file
    :return: |retfuncfails0|

.. include:: calceph.time_set_relationship_tt_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tt|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tdb_to_jd_tt(t_calcephbin * eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tt, double *jdfrac_tt)

    :param  eph: |arg_eph|
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :param jd0_tt: Integer part of Julian Date (TT)
    :param jdfrac_tt: Fractional part of Julian Date (TT)
 
    :return: |retfuncfails0|

.. include:: calceph.time_tdb_to_tt.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tt_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tt_to_jd_tdb(t_calcephbin * eph, double jd0_tt, double jdfrac_tt, double *jd0_tdb, double *jdfrac_tdb)

    :param  eph: |arg_eph|
    :param jd0_tt: Integer part of Julian Date (TT)
    :param jdfrac_tt: Fractional part of Julian Date (TT)
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
 
    :return: |retfuncfails0|

.. include:: calceph.time_tt_to_tdb.rst


.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tcb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tdb_to_jd_tcb(t_calcephbin * eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tcb, double *jdfrac_tcb)

    :param  eph: |arg_eph|
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :param jd0_tcb: Integer part of Julian Date (TCB)
    :param jdfrac_tcb: Fractional part of Julian Date (TCB)
 
    :return: |retfuncfails0|

.. include:: calceph.time_tdb_to_tcb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tcb_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tcb_to_jd_tdb(t_calcephbin * eph, double jd0_tcb, double jdfrac_tcb, double *jd0_tdb, double *jdfrac_tdb)

    :param  eph: |arg_eph|
    :param jd0_tcb: Integer part of Julian Date (TCB)
    :param jdfrac_tcb: Fractional part of Julian Date (TCB)
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
 
    :return: |retfuncfails0|

.. include:: calceph.time_tcb_to_tdb.rst


.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_cal_utc_to_jd_tdb(t_calcephbin * eph, int yy, int month, int day, int hh, int min, double sec, double *jd0_tdb, double *jdfrac_tdb)

   :param  eph: |arg_eph|
   :param yy: Year
   :param month: Month
   :param day: Day
   :param hh: Hour
   :param min: Minute
   :param sec: Second
   :param jd0_tdb: Integer part of the resulting TDB Julian Date
   :param jdfrac_tdb: Fractional part of the resulting TDB Julian Date

   :return: |retfuncfails0|

.. include:: calceph.time_cal_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_cal_utc|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tdb_to_cal_utc(t_calcephbin * eph, double jd0_tdb, double jdfrac_tdb, int *yy, int *month, int *day, int *hh, int *min, double *sec)

   :param  eph: |arg_eph|
   :param jd0_tdb: Integer part of the TDB Julian Date
   :param jdfrac_tdb: Fractional part of the TDB Julian Date
   :param yy: Year
   :param month: Month
   :param day: Day
   :param hh: Hour
   :param min: Minute
   :param sec: Second

   :return: |retfuncfails0|

.. include:: calceph.time_jd_tdb_to_cal_utc.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_str_utc_to_jd_tdb(t_calcephbin * eph, const char *str, double *jd0_tdb, double *jdfrac_tdb)

    :param  eph: |arg_eph|
    :param str: UTC time string to parse
    :param jd0_tdb: Integer part of the resulting TDB Julian Date
    :param jdfrac_tdb: Fractional part of the resulting TDB Julian Date

    :return: |retfuncfails0|

.. include:: calceph.time_str_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_any_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_str_any_to_jd_tdb(t_calcephbin * eph, const char *str, double *jd0_tdb, double *jdfrac_tdb)

   :param  eph: |arg_eph|
   :param str: Time string to parse
   :param jd0_tdb: Integer part of the resulting TDB Julian Date
   :param jdfrac_tdb: Fractional part of the resulting TDB Julian Date

   :return: |retfuncfails0|

.. include:: calceph.time_str_any_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_spacecraft_clock_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_str_spacecraft_clock_to_jd_tdb(t_calcephbin * eph, int target, const char *str, double *jd0_tdb, double *jdfrac_tdb)

    :param  eph: |arg_eph|
    :param target: NAIF ID of the spacecraft
    :param str: Spacecraft clock string (e.g., "1/1234:56")
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)

    :return: |retfuncfails0|

.. include:: calceph.time_str_sclk_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_str_spacecraft_clock|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_time_jd_tdb_to_str_spacecraft_clock(t_calcephbin * eph, int target, double jd0_tdb, double jdfrac_tdb, t_calcephcharvalue str)

    :param  eph: |arg_eph|
    :param target: NAIF ID of the spacecraft
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :param str: Output string buffer

    :return: |retfuncfails0|

.. include:: calceph.time_jd_tdb_to_str_sclk.rst

