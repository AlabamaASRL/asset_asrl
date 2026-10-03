.. |arg_eph|  replace:: ephemeris descriptor
.. |arg_filename|  replace:: pathname of the file
.. |arg_ifname|  replace:: internal filename (max 60 characters)
.. |arg_flags|  replace:: reserved for future extensions (must be zero)
.. |arg_n|  replace:: number of files
.. |arg_array_filename|  replace:: array of pathname of the files
.. |arg_len_filename|  replace:: number of characters of each file's name
.. |arg_JD0|  replace:: Integer part of the Julian date (TDB or TCB)
.. |arg_time|  replace:: Fraction part of the Julian date (TDB or TCB)
.. |arg_target|  replace:: The body or reference point whose coordinates are required (see the list, below).
.. |arg_center|  replace:: The origin of the coordinate system (see the list, below). If *target* is 14, 15, 16 or 17 (nutation, libration, TT-TDB or TCG-TCB), *center* must be *0*.
.. |arg_target_unit|  replace:: The body or reference point whose coordinates are required. The numbering system depends on the parameter unit. 
.. |arg_center_unit|  replace:: The origin of the coordinate system. The numbering system depends on the parameter unit.
.. |arg_target_orient_unit|  replace:: The body whose orientations are requested. The numbering system depends on the parameter unit. 
.. |arg_constant_name|  replace:: name of the constant
.. |arg_constant_value|  replace:: first value of the constant
.. |arg_constant_arrayvalue|  replace:: array of values for the constant
.. |arg_constant_nvalue|  replace:: number of elements of the array 
.. |arg_constant_number| replace:: number of constants
.. |arg_constant_index| replace:: index of the constant, between 1 and |calceph_getconstantcount|
.. |arg_version| replace:: version of the library
.. |arg_typehandler| replace:: type of handler
.. |arg_userfunc| replace:: user function
.. |arg_firsttime|  replace:: Julian date of the first time
.. |arg_lasttime|  replace:: Julian date of the last time 
.. |arg_continuous|  replace:: information about the availability of the quantities over the time span
.. |arg_positionrecordcount| replace:: number of position's records
.. |arg_positionrecord_target|  replace:: The target body
.. |arg_positionrecord_center|  replace:: The origin body
.. |arg_positionrecord_index| replace:: index of the position's record, between 1 and |calceph_getpositionrecordcount|
.. |arg_orientrecordcount| replace:: number of orientation's records
.. |arg_orientrecord_index| replace:: index of the orientation's record, between 1 and |calceph_getorientrecordcount|
.. |arg_frame| replace:: reference frame (see the list, below)
.. |arg_fileversion| replace:: version of the file
.. |arg_segid| replace:: type of the segment.
.. |timescale_JD0_Time| replace:: The date (JD0, time) should be expressed in the same time scale as the ephemeris files, which can be retrieved using the function |calceph_gettimescale|.  
.. |warning_UTC| replace:: If a date, expressed in the Coordinated Universal Time (UTC), is supplied to this function, a very large erroneous position will be returned. The function |calceph_time_cal_utc_to_jd_tdb| can convert the UTC calendar date to the Julian Day expressed in the time scale TDB. 
.. |arg_name_getidbyname|  replace:: name of the body
.. |arg_unit_getidbyname|  replace:: 0 or CALCEPH_USE_NAIFID
.. |arg_id_getidbyname|  replace:: id of the body. The numbering system depends on the parameter unit.
.. |arg_write_comment|  replace:: comment to be written.
.. |arg_write_target|  replace:: The body or reference point whose coordinates will be written (see :ref:`NAIF identification numbers`).
.. |arg_write_targets| replace:: array of bodies or reference points whose coordinates will be written (see :ref:`NAIF identification numbers`).
.. |arg_write_center|  replace:: The origin of the coordinate system (see :ref:`NAIF identification numbers`).
.. |arg_write_frame| replace:: reference frame (see :ref:`Frame numbers`).
.. |arg_write_start_jd0_tdb|  replace:: starting Julian date integer part (TDB).
.. |arg_write_start_frac_tdb|  replace:: starting Julian date fraction part (TDB).
.. |arg_write_end_jd0_tdb|  replace:: stop Julian date integer part (TDB).
.. |arg_write_end_frac_tdb|  replace:: stop Julian date fraction part (TDB).
.. |arg_write_intlen_jd_tdb|  replace:: length of the interpolation interval (in days, TDB).
.. |arg_write_intlens_jd_tdb| replace:: array containing the interpolation interval length for each target (in days, TDB).
.. |arg_write_start_jd0_tcb|  replace:: starting Julian date integer part (TCB).
.. |arg_write_start_frac_tcb|  replace:: starting Julian date fraction part (TCB).
.. |arg_write_end_jd0_tcb|  replace:: stop Julian date integer part (TCB).
.. |arg_write_end_frac_tcb|  replace:: stop Julian date fraction part (TCB).
.. |arg_write_intlen_jd_tcb|  replace:: length of the interpolation interval (in days, TCB).
.. |arg_write_intlens_jd_tcb| replace:: array containing the interpolation interval length for each target (in days, TCB).
.. |arg_write_polynomials|  replace:: array of Chebyshev polynomial coefficients.
.. |arg_write_states|  replace:: array of state vectors.
.. |arg_write_data| replace:: array data to be written.
.. |arg_write_epochs|  replace:: array of epochs (in Julian date TCB) one per state vector.
.. |arg_write_record_count|  replace:: number of records to be written.
.. |arg_write_record_counts| replace:: array of number of records for each target.
.. |arg_write_poly_deg|  replace:: degree of the Chebyshev polynomials.
.. |arg_write_inter_deg|  replace:: degree of interpolation.
.. |arg_write_segid| replace:: segment identifier (must be unique within the file, max 40 characters).
.. |arg_write_segids| replace:: array of segment identifiers (one per target, each must be unique within the file, max 40 characters).
.. |arg_write_target_count| replace:: number of targets the reservation is done for.
.. |arg_write_reservation|  replace:: reservation identifier.
.. |arg_write_target_index|  replace:: index of the target body in the reservation.
.. |arg_write_record_begin_index|  replace:: index of the first record to be written in the target's segment.
