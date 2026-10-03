.. include:: replace.rst


|menu_writeph_spk_create|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk_create(filename, ifname, flags) BIND(C)

    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :param ifname [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_ifname|
    :param flags [INTEGER(C_INT), VALUE, intent(in))]: |arg_flags|
    :rtype writeph_spk_create: TYPE(C_PTR)


.. include:: writeph.spk_create.rst

.. %------------------------------------------------

|menu_writeph_spk_open|
~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk_open(filename) BIND(C)

    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :rtype writeph_spk_open: TYPE(C_PTR)

.. include:: writeph.spk_open.rst

.. %----------------------------------------------------------------------------

|menu_writeph_close|
~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_close(eph) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :r writeph_close: |retfuncfails0|
    :rtype writeph_close: INTEGER(C_INT)

.. include:: writeph.spk_close.rst

.. %----------------------------------------------------------------------------

|menu_writeph_dump|
~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_dump(eph, filename) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :return writeph_dump: |retfuncfails0|
    :rtype writeph_dump: INTEGER(C_INT)

.. include:: writeph.spk_dump.rst


.. %----------------------------------------------------------------------------

|menu_writeph_restore|
~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_restore(eph, filename) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|
    :return writeph_restore: |retfuncfails0|
    :rtype writeph_restore: INTEGER(C_INT)

.. include:: writeph.spk_restore.rst


.. %----------------------------------------------------------------------------

|menu_writeph_comment|
~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_comment(eph, comment) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param comment [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_comment|
    :return writeph_comment: |retfuncfails0|
    :rtype writeph_comment: INTEGER(C_INT)

.. include:: writeph.spk_comment.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk2_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk2_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
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
    :return writeph_spk2_seq_write: |retfuncfails0|
    :rtype writeph_spk2_seq_write: INTEGER(C_INT)

.. include:: writeph.spk2_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk2_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk2_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk2_par_reserve: |retreservation|
    :rtype writeph_spk2_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk2_par_reserve.rst


.. include:: replace.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk2_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk2_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_spk2_par_write: |retfuncfails0|
    :rtype writeph_spk2_par_write: INTEGER(C_INT)

.. include:: writeph.spk2_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk3_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk3_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
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
    :return writeph_spk3_seq_write: |retfuncfails0|
    :rtype writeph_spk3_seq_write: INTEGER(C_INT)

.. include:: writeph.spk3_seq_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk3_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk3_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk3_par_reserve: |retreservation|
    :rtype writeph_spk3_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk3_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk3_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk3_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_spk3_par_write: |retfuncfails0|
    :rtype writeph_spk3_par_write: INTEGER(C_INT)

.. include:: writeph.spk3_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk8_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk8_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tdb|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_spk8_seq_write: |retfuncfails0|
    :rtype writeph_spk8_seq_write: INTEGER(C_INT)

.. include:: writeph.spk8_seq_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk8_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk8_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb [REAL(C_DOUBLE), dimension(*), intent(in)]:  |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk8_par_reserve: |retreservation|
    :rtype writeph_spk8_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk8_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk8_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk8_par_write(eph, reservation, target_index, record_begin_index, record_count, states) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :return writeph_spk8_par_write: |retreservation|
    :rtype writeph_spk8_par_write: INTEGER(C_INT)

.. include:: writeph.spk8_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk9_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. f:function:: function writeph_spk9_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, states, epochs, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :param epochs [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_epochs|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_spk8_par_write: |retfuncfails0|
    :rtype writeph_spk8_par_write: INTEGER(C_INT)

.. include:: writeph.spk9_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk9_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk9_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk9_par_reserve: |retreservation|
    :rtype writeph_spk9_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk9_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk9_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk9_par_write(eph, reservation, target_index, record_begin_index, record_count, states, epochs) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :param epochs [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_epochs|
    :return writeph_spk9_par_write: |retfuncfails0|
    :rtype writeph_spk9_par_write: INTEGER(C_INT)

.. include:: writeph.spk9_par_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk12_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk12_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, states, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tdb|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_spk12_seq_write: |retfuncfails0|
    :rtype writeph_spk12_seq_write: INTEGER(C_INT)

.. include:: writeph.spk12_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk12_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk12_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb [REAL(C_DOUBLE), dimension(*), intent(in)]:  |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk12_par_reserve: |retreservation|
    :rtype writeph_spk12_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk12_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk12_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk12_par_write(eph, reservation, target_index, record_begin_index, record_count, states) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :return writeph_spk12_par_write: |retfuncfails0|
    :rtype writeph_spk12_par_write: INTEGER(C_INT)

.. include:: writeph.spk12_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk13_seq_write(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, states, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlen_jd_tdb|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_spk13_seq_write: |retfuncfails0|
    :rtype writeph_spk13_seq_write: INTEGER(C_INT)

.. include:: writeph.spk13_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk13_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb [REAL(C_DOUBLE), dimension(*), intent(in)]:  |arg_write_intlens_jd_tdb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_inter_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk13_par_reserve: |retreservation|
    :rtype writeph_spk13_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk13_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk13_par_write(eph, reservation, target_index, record_begin_index, record_count, states) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param states [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_states|
    :return writeph_spk13_par_write: |retfuncfails0|
    :rtype writeph_spk13_par_write: INTEGER(C_INT)

.. include:: writeph.spk13_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk14_begin|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk14_begin(eph, target, center, frame, start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tdb|
    :param start_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tdb|
    :param end_jd0_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tdb|
    :param end_frac_tdb [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tdb|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segid [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_write_segid|
    :return writeph_spk14_begin: |retfuncfails0|
    :rtype writeph_spk14_begin: INTEGER(C_INT)

.. include:: writeph.spk14_begin.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk14_add|
~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk14_add(eph, record_count, data, epochs) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param data [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_data|
    :param epochs [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_epochs|
    :return writeph_spk14_add: |retfuncfails0|
    :rtype writeph_spk14_add: INTEGER(C_INT)

.. include:: writeph.spk14_add.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk14_end|
~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk14_end(eph) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :return writeph_spk14_end: |retfuncfails0|
    :rtype writeph_spk14_end: INTEGER(C_INT)

.. include:: writeph.spk14_end.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk102_seq_write(eph, target, center, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlen_jd_tcb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
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
    :return writeph_spk102_seq_write: |retfuncfails0|
    :rtype writeph_spk102_seq_write: INTEGER(C_INT)

.. include:: writeph.spk102_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk102_par_reserve(eph, target_count, targets, center, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlens_jd_tcb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk102_par_reserve: |retreservation|
    :rtype writeph_spk102_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk102_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk102_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_spk102_par_write: |retfuncfails0|
    :rtype writeph_spk102_par_write: INTEGER(C_INT)

.. include:: writeph.spk102_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk103_seq_write(eph, target, center, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlen_jd_tcb, polynomials, record_count, deg, segid) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
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
    :return writeph_spk103_seq_write: |retfuncfails0|
    :rtype writeph_spk103_seq_write: INTEGER(C_INT)

.. include:: writeph.spk103_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk103_par_reserve(eph, target_count, targets, center, frame, start_jd0_tcb, start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb, record_counts, deg, segids)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param target_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_count|
    :param targets [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_targets|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_center|
    :param frame [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_frame|
    :param start_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_jd0_tcb|
    :param start_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_start_frac_tcb|
    :param end_jd0_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_jd0_tcb|
    :param end_frac_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb  [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_write_intlens_jd_tcb|
    :param record_counts [INTEGER(C_INT), dimension(*), intent(in)]: |arg_write_record_counts|
    :param deg [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_poly_deg|
    :param segids [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_write_segids|
    :return writeph_spk103_par_reserve: |retreservation|
    :rtype writeph_spk103_par_reserve: INTEGER(C_INT)

.. include:: writeph.spk103_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: function writeph_spk103_par_write(eph, reservation, target_index, record_begin_index, record_count, polynomials) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param reservation [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_reservation|
    :param target_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_target_index|
    :param record_begin_index [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_begin_index|
    :param record_count [INTEGER(C_INT), VALUE, intent(in)]: |arg_write_record_count|
    :param polynomials [REAL(C_DOUBLE), dimension(*), intent(in)]: |arg_write_polynomials|
    :return writeph_spk103_seq_write: |retfuncfails0|
    :rtype writeph_spk103_seq_write: INTEGER(C_INT)

.. include:: writeph.spk103_par_write.rst
