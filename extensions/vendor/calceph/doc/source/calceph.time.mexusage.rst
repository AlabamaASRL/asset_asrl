.. _`Time functions`:

.. include:: calceph.time_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_time_jd_to_cal|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_to_cal (timescale, jd0, jdfrac) -> yy, month, day, hh, min, sec

    :param int timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param int jd0: Integer part of Julian Date
    :param int jdfrac: Fractional part of Julian Date
    :return:  Year, Month [1–12], Day [1–31], Hour [0–23], Minute [0–59], seconds [0.0–61.0[
    :rtype: int, int, int, int, int, float

.. include:: calceph.time_jd_to_cal.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_cal_to_jd(timescale, yy, month, day, hh, min, sec) -> jd0, jdfrac

    :param int timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param int yy: Year
    :param int month: Month [1–12]
    :param int day: Day [1–31]
    :param int hh: Hour [0–23]
    :param int min: Minute [0–59]
    :param float sec: Seconds [0.0–61.0[
    :return: Integer part of Julian Date, Fractional part of Julian Date
    :rtype: float, float

.. include:: calceph.time_cal_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_to_jd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_str_to_jd(timescale, str) -> jd0, jdfrac

    :param timescale: timescale of the input date and result ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    :param str: Input time string to be parsed
    :return: Integer part of Julian Date, Fractional part of Julian Date
    :rtype: float, float

.. include:: calceph.time_str_to_jd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_set_relationship_tt_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_set_relationship_tt_tdb(model)

    :param model: | Relationship model
                  | 0: use of |calceph_compute_unit|
                  | 1: use the model based on TLS file

.. include:: calceph.time_set_relationship_tt_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tt|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tdb_to_jd_tt(jd0_tdb, jdfrac_tdb) -> jd0_tt, jdfrac_tt

    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :return: Integer part of Julian Date (TT), Fractional part of Julian Date (TT)
    :rtype: float, float


.. include:: calceph.time_tdb_to_tt.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tt_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tt_to_jd_tdb(jd0_tt, jdfrac_tt) -> jd0_tdb, jdfrac_tdb

    :param jd0_tt: Integer part of Julian Date (TT)
    :param jdfrac_tt: Fractional part of Julian Date (TT)
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float
 
.. include:: calceph.time_tt_to_tdb.rst


.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_jd_tcb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tdb_to_jd_tcb(jd0_tdb, jdfrac_tdb) -> jd0_tcb, jdfrac_tcb

    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :return: Integer part of Julian Date (TCB), Fractional part of Julian Date (TCB)
    :rtype: float, float


.. include:: calceph.time_tdb_to_tcb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tcb_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tcb_to_jd_tdb(jd0_tcb, jdfrac_tcb) -> jd0_tdb, jdfrac_tdb

    :param jd0_tcb: Integer part of Julian Date (TCB)
    :param jdfrac_tcb: Fractional part of Julian Date (TCB)
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float
 
.. include:: calceph.time_tcb_to_tdb.rst


.. %----------------------------------------------------------------------------

|menu_calceph_time_cal_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_cal_utc_to_jd_tdb(yy, month, day, hh, min, sec) -> jd0_tdb, jdfrac_tdb

    :param yy: Year
    :param month: Month
    :param day: Day
    :param hh: Hour
    :param min: Minute
    :param sec: Second
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float

   :return: 0 on success, 1 on error

.. include:: calceph.time_cal_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_cal_utc|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tdb_to_cal_utc( jd0_tdb,  jdfrac_tdb) -> yy, month, day, hh, min, sec

    :param jd0_tdb: Integer part of the TDB Julian Date
    :param jdfrac_tdb: Fractional part of the TDB Julian Date
    :return:  Year, Month [1–12], Day [1–31], Hour [0–23], Minute [0–59], seconds [0.0–61.0[
    :rtype: int, int, int, int, int, float

.. include:: calceph.time_jd_tdb_to_cal_utc.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_utc_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_str_utc_to_jd_tdb(str) -> jd0_tdb, jdfrac_tdb

    :param str: UTC time string to parse
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float

.. include:: calceph.time_str_utc_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_any_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_str_any_to_jd_tdb(str) -> jd0_tdb, jdfrac_tdb

    :param str: Time string to parse
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float

.. include:: calceph.time_str_any_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_str_spacecraft_clock_to_jd_tdb|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_str_spacecraft_clock_to_jd_tdb(target, strdate) -> jd0_tdb, jdfrac_tdb

    :param target: NAIF ID of the spacecraft
    :param strdate: Spacecraft clock string (e.g., "1/1234:56")
    :return: Integer part of Julian Date (TDB), Fractional part of Julian Date (TDB)
    :rtype: float, float

.. include:: calceph.time_str_sclk_to_jd_tdb.rst

.. %----------------------------------------------------------------------------

|menu_calceph_time_jd_tdb_to_str_spacecraft_clock|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.time_jd_tdb_to_str_spacecraft_clock(eph, target, jd0_tdb,  jdfrac_tdb) 

    :param  eph: |arg_eph|
    :param target: NAIF ID of the spacecraft
    :param jd0_tdb: Integer part of Julian Date (TDB)
    :param jdfrac_tdb: Fractional part of Julian Date (TDB)
    :return: Output spacecraft clock time string
    :rtype: str

    :return: |retfuncfails0|

.. include:: calceph.time_jd_tdb_to_str_sclk.rst
