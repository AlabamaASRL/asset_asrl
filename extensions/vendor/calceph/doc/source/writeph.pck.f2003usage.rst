
.. include:: replace.rst



|menu_writeph_pck_create|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck_create(filename, ifname, flags) BIND(C)

    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :param ifname [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_ifname|
    :param flags [INTEGER(C_INT), VALUE, intent(in))]: |arg_flags|
    :rtype writeph_pck_create: TYPE(C_PTR)


.. include:: writeph.pck_create.rst

.. %------------------------------------------------

|menu_writeph_pck_open|
~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck_open(filename) BIND(C)

    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :rtype writeph_pck_open: TYPE(C_PTR)

.. include:: writeph.pck_open.rst


The following PCK functions are to be used to write binary PCK files:

.. %----------------------------------------------------------------------------

|menu_writeph_pck2_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck2_seq_write(eph, target, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tdb|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_pck2_seq_write: |retfuncfails0|
    :rtype writeph_pck2_seq_write: INTEGER(C_INT)

.. include:: writeph.pck2_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck2_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck2_par_reserve(eph, target_count, targets, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_pck2_par_reserve: |retreservation|
    :rtype writeph_pck2_par_reserve: INTEGER(C_INT)

.. include:: writeph.pck2_par_reserve.rst


.. %----------------------------------------------------------------------------

|menu_writeph_pck2_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck2_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_pck2_par_write: |retfuncfails0|
    :rtype writeph_pck2_par_write: INTEGER(C_INT)

.. include:: writeph.pck2_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck3_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck3_seq_write(eph, target, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tdb|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_pck3_seq_write: |retfuncfails0|
    :rtype writeph_pck3_seq_write: INTEGER(C_INT)

.. include:: writeph.pck3_seq_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_pck3_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck3_par_reserve(eph, target_count, targets, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_pck3_par_reserve: |retreservation|
    :rtype writeph_pck3_par_reserve: INTEGER(C_INT)

.. include:: writeph.pck3_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck3_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck3_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_pck3_par_write: |retfuncfails0|
    :rtype writeph_pck3_par_write: INTEGER(C_INT)

.. include:: writeph.pck3_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck102_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck102_seq_write(eph, target, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlen_jd_tcb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlen_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tcb|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_pck102_seq_write: |retfuncfails0|
    :rtype writeph_pck102_seq_write: INTEGER(C_INT)

.. include:: writeph.pck102_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck102_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck102_par_reserve(eph, target_count, targets, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlens_jd_tcb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_pck102_par_reserve: |retreservation|
    :rtype writeph_pck102_par_reserve: INTEGER(C_INT)

.. include:: writeph.pck102_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck102_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck102_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_pck102_par_write: |retfuncfails0|
    :rtype writeph_pck102_par_write: INTEGER(C_INT)

.. include:: writeph.pck102_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck103_seq_write(eph, target, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlen_jd_tcb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlen_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tcb|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_pck103_seq_write: |retfuncfails0|
    :rtype writeph_pck103_seq_write: INTEGER(C_INT)

.. include:: writeph.pck103_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck103_par_reserve(eph, target_count, targets, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlens_jd_tcb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_pck103_par_reserve: |retreservation|
    :rtype writeph_pck103_par_reserve: INTEGER(C_INT)

.. include:: writeph.pck103_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_pck103_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_pck103_seq_write: |retfuncfails0|
    :rtype writeph_pck103_seq_write: INTEGER(C_INT)

.. include:: writeph.pck103_par_write.rst
