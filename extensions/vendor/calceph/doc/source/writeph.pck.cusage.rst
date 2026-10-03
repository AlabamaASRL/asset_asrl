

.. include:: replace.rst


|menu_writeph_pck_create|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: t_writephbin* writeph_pck_create(const char *filename, const char *ifname, int flags)

    :param filename: |arg_filename|
    :param ifname: |arg_ifname|
    :param flags: |arg_flags|
    :return: |arg_eph|. |retfuncfailsNULL|

.. include:: writeph.pck_create.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck_open|
~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: t_writephbin* writeph_pck_open(const char *filename)

    :param filename: |arg_filename|
    :return: |arg_eph|. |retfuncfailsNULL|



.. include:: writeph.pck_open.rst



The following PCK functions are to be used to write binary PCK files:


.. %----------------------------------------------------------------------------

|menu_writeph_pck2_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck2_seq_write(t_writephbin *eph, int target, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
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

.. include:: writeph.pck2_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck2_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck2_par_reserve(t_writephbin *eph, int target_count, const int *targets, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
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

.. include:: writeph.pck2_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck2_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck2_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.pck2_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck3_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck3_seq_write(t_writephbin *eph, int target, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
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

.. include:: writeph.pck3_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck3_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck3_par_reserve(t_writephbin *eph, int target_count, const int *targets, int frame, double start_jd0_tdb, double start_frac_tdb, double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
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

.. include:: writeph.pck3_par_reserve.rst


.. %----------------------------------------------------------------------------

|menu_writeph_pck3_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck3_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.pck3_par_write.rst


.. %----------------------------------------------------------------------------

|menu_writeph_pck102_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck102_seq_write(t_writephbin *eph, int target, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, double intlen_jd_tcb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
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

.. include:: writeph.pck102_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck102_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck102_par_reserve(t_writephbin *eph, int target_count, const int *targets, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|

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

.. include:: writeph.pck102_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck102_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck102_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.pck102_par_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_seq_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck103_seq_write(t_writephbin *eph, int target, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, double intlen_jd_tcb, const double *polynomials, int record_count, int deg, const char *segid)

    :param eph: |arg_eph|
    :param target: |arg_write_target|
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

.. include:: writeph.pck103_seq_write.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_par_reserve|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck103_par_reserve(t_writephbin *eph, int target_count, const int *targets, int frame, double start_jd0_tcb, double start_frac_tcb, double end_jd0_tcb, double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids[])

    :param eph: |arg_eph|
    :param target_count: |arg_write_target_count|
    :param targets: |arg_write_targets|
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

.. include:: writeph.pck103_par_reserve.rst

.. %----------------------------------------------------------------------------

|menu_writeph_pck103_par_write|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int writeph_pck103_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index, int record_count, const double *polynomials)

    :param eph: |arg_eph|
    :param reservation: |arg_write_reservation|
    :param target_index: |arg_write_target_index|
    :param record_begin_index: |arg_write_record_begin_index|
    :param record_count: |arg_write_record_count|
    :param polynomials: |arg_write_polynomials|
    :return: |retfuncfails0|

.. include:: writeph.pck103_par_write.rst
