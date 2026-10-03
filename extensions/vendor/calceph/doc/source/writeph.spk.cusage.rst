

.. include:: replace.rst
    

|menu_writeph_spk_create|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: t_writephbin* writeph_spk_create(const char *filename, const char *ifname, int flags)

    :param filename: |arg_filename|
    :param ifname: |arg_ifname|
    :param flags: |arg_flags|
    :return: |arg_eph|. |retfuncfailsNULL|

.. include:: writeph.spk_create.rst



|menu_writeph_spk_open|
~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: t_writephbin* writeph_spk_open(const char *filename)

    :param filename: |arg_filename|
    :return: |arg_eph|. |retfuncfailsNULL|

.. include:: writeph.spk_open.rst

.. %----------------------------------------------------------------------------

|menu_writeph_close|
~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_close(t_writephbin *eph)

    :param eph: |arg_eph|
    :return: |retfuncfails0|


.. include:: writeph.spk_close.rst

.. %----------------------------------------------------------------------------

|menu_writeph_dump|
~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_dump(t_writephbin *eph, const char *filename)

    :param eph: |arg_eph|
    :param filename: |arg_filename|
    :return: |retfuncfails0|

.. include:: writeph.spk_dump.rst


.. %----------------------------------------------------------------------------

|menu_writeph_restore|
~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_restore(t_writephbin *eph, const char *filename)

    :param eph: |arg_eph|
    :param filename: |arg_filename|
    :return: |retfuncfails0|

.. include:: writeph.spk_restore.rst

.. %----------------------------------------------------------------------------

|menu_writeph_comment|
~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_comment(t_writephbin *eph, const char *comment)

    :param eph: |arg_eph|
    :param comment: |arg_write_comment|
    :return: |retfuncfails0|

.. include:: writeph.spk_comment.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk2_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk2_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb: |arg_write_intlen_jd_tdb|
    :param polynomials: |arg_write_polynomials|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_poly_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk2_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk2_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk2_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_poly_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk2_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk2_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk2_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.spk2_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk3_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk3_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb: |arg_write_intlen_jd_tdb|
    :param polynomials: |arg_write_polynomials|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_poly_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk3_seq_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk3_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk3_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_poly_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk3_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk3_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk3_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.spk3_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk8_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk8_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb: |arg_write_intlen_jd_tdb|
    :param states: |arg_write_states|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_inter_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk8_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk8_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk8_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_inter_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk8_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk8_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk8_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *states)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param states: |arg_write_states|
    :return: |retfuncfails0|

.. include:: writeph.spk8_par_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk9_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. c:function:: int writeph_spk9_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *states, const double *epochs, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param states: |arg_write_states|
    :param epochs: |arg_write_epochs|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_inter_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk9_seq_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk9_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk9_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_inter_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk9_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk9_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk9_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *states, const double *epochs)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param states: |arg_write_states|
    :param epochs: |arg_write_epochs|
    :return: |retfuncfails0|

.. include:: writeph.spk9_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk12_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk12_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *states, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlen_jd_tdb: |arg_write_intlen_jd_tdb|
    :param states: |arg_write_states|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_inter_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk12_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk12_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk12_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param intlens_jd_tdb: |arg_write_intlens_jd_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_inter_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk12_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk12_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk12_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *states)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param states: |arg_write_states|
    :return: |retfuncfails0|

.. include:: writeph.spk12_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk13_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *states, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param states: |arg_write_states|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_inter_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk13_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk13_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_inter_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk13_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk13_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk13_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *states, const double *epochs)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param states: |arg_write_states|
    :param epochs: |arg_write_epochs|
    :return: |retfuncfails0|

.. include:: writeph.spk13_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk14_begin|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk14_begin(t_writephbin *eph, int target, int center, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tdb: |arg_write_start_jd0_tdb|
    :param start_frac_tdb: |arg_write_start_frac_tdb|
    :param end_jd0_tdb: |arg_write_end_jd0_tdb|
    :param end_frac_tdb: |arg_write_end_frac_tdb|
    :param deg: |arg_write_poly_deg|
    :param segid: |arg_write_segid|

.. include:: writeph.spk14_begin.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk14_add|
~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk14_add(t_writephbin *eph, int record_count, const double *data, const double *epochs)

    :param eph: |arg_eph|
    :param record_count: |arg_write_record_count|
    :param data: |arg_write_data|
    :param epochs: |arg_write_epochs|

.. include:: writeph.spk14_add.rst


.. %----------------------------------------------------------------------------

|menu_writeph_spk14_end|
~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk14_end(t_writephbin *eph)

    :param eph: |arg_eph|

.. include:: writeph.spk14_end.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk102_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, double intlen_jd_tcb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tcb: |arg_write_start_jd0_tcb|
    :param start_frac_tcb: |arg_write_start_frac_tcb|
    :param end_jd0_tcb: |arg_write_end_jd0_tcb|
    :param end_frac_tcb: |arg_write_end_frac_tcb|
    :param intlen_jd_tcb: |arg_write_intlen_jd_tcb|
    :param polynomials: |arg_write_polynomials|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_poly_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk102_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk102_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tcb: |arg_write_start_jd0_tcb|
    :param start_frac_tcb: |arg_write_start_frac_tcb|
    :param end_jd0_tcb: |arg_write_end_jd0_tcb|
    :param end_frac_tcb: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb: |arg_write_intlens_jd_tcb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_poly_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk102_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk102_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk102_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.spk102_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk103_seq_write(t_writephbin *eph, int target, int center, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, double intlen_jd_tcb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tcb: |arg_write_start_jd0_tcb|
    :param start_frac_tcb: |arg_write_start_frac_tcb|
    :param end_jd0_tcb: |arg_write_end_jd0_tcb|
    :param end_frac_tcb: |arg_write_end_frac_tcb|
    :param intlen_jd_tcb: |arg_write_intlen_jd_tcb|
    :param polynomials: |arg_write_polynomials|
    :param record_count: |arg_write_record_count|
    :param deg: |arg_write_poly_deg|
    :param segid: |arg_write_segid|
    :return: |retfuncfails0|

.. include:: writeph.spk103_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk103_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
    :param center: |arg_write_center|
    :param frame: |arg_write_frame|
    :param start_jd0_tcb: |arg_write_start_jd0_tcb|
    :param start_frac_tcb: |arg_write_start_frac_tcb|
    :param end_jd0_tcb: |arg_write_end_jd0_tcb|
    :param end_frac_tcb: |arg_write_end_frac_tcb|
    :param intlens_jd_tcb: |arg_write_intlens_jd_tcb|
    :param record_counts: |arg_write_record_counts|
    :param deg: |arg_write_poly_deg|
    :param segids: |arg_write_segids|
    :return: |retreservation|

.. include:: writeph.spk103_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_spk103_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_spk103_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.spk103_par_write.rst
