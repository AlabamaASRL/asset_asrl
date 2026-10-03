.. _`Reading/Evaluation functions`:

.. include:: calceph.multiple_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_open|
~~~~~~~~~~~~~~~~~~~


.. f:function:: function calceph_open (filename) BIND(C)

    :p  filename [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_filename|.
    :r calceph_open: |arg_eph|. |retfuncfailsNULL|
    :rtype calceph_open: TYPE(C_PTR)


.. include:: calceph.multiple_open.rst

.. %----------------------------------------------------------------------------

|menu_calceph_open_array|
~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: function calceph_open_array (n, array_filename, len_filename) BIND(C)

    :p  n [INTEGER(C_INT), VALUE, intent(in)]: |arg_n|.
    :p  array_filename [CHARACTER(len=1,kind=C_CHAR), dimension(*), intent(in)]: |arg_array_filename|.
    :p  len_filename [INTEGER(C_INT), VALUE, intent(in)]: |arg_len_filename|.
    :r calceph_open_array: |arg_eph|. |retfuncfailsNULL|
    :rtype calceph_open_array: TYPE(C_PTR)


.. include:: calceph.multiple_open_array.rst

.. %----------------------------------------------------------------------------

|menu_calceph_prefetch|
~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_prefetch (eph) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :r calceph_prefetch: |retfuncfails0|
    :rtype calceph_prefetch: INTEGER(C_INT)


.. include:: calceph.multiple_prefetch.rst

.. %----------------------------------------------------------------------------

|menu_calceph_isthreadsafe|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_isthreadsafe (eph) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :r calceph_isthreadsafe: returns 1 if multiple threads can access the same ephemeris ephemeris descriptor, otherwise 0.
    :rtype calceph_isthreadsafe: INTEGER(C_INT)


.. include:: calceph.multiple_isthreadsafe.rst


.. %----------------------------------------------------------------------------

|menu_calceph_compute|
~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_compute (eph, JD0, time, target, center, PV ) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_center|
    :param PV[REAL(C_DOUBLE), dimension(1\:6), intent(out)]: .. include:: arg_PV.rst
    :r calceph_compute: |retfuncfails0|
    :rtype calceph_compute: INTEGER(C_INT)


.. include:: calceph.multiple_compute.rst

   
.. %----------------------------------------------------------------------------

|menu_calceph_compute_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_compute_unit (eph, JD0, time, target, center, unit, PV ) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target_unit|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_center_unit|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit.rst
    :param PV[REAL(C_DOUBLE), dimension(1\:6), intent(out)]: .. include:: arg_PV_unit.rst
    :r calceph_compute_unit: |retfuncfails0|
    :rtype calceph_compute_unit: INTEGER(C_INT)
    

.. include:: calceph.multiple_compute_unit.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_orient_unit (eph, JD0, time, target, unit, PV) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param PV[REAL(C_DOUBLE), dimension(1\:6), intent(out)]: .. include:: arg_PV_orient_unit.rst
    :r calceph_orient_unit: |retfuncfails0|
    :rtype calceph_orient_unit: INTEGER(C_INT)

.. include:: calceph.multiple_orient_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_rotangmom_unit (eph, JD0, time, target, unit, PV) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param PV[REAL(C_DOUBLE), dimension(1\:6), intent(out)]: .. include:: arg_PV_rotangmom_unit.rst
    :r calceph_rotangmom_unit: |retfuncfails0|
    :rtype calceph_rotangmom_unit: INTEGER(C_INT)

.. include:: calceph.multiple_rotangmom_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_compute_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_compute_order (eph, JD0, time, target, center, unit, order, PVAJ ) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target|
    :param center [INTEGER(C_INT), VALUE, intent(in)]: |arg_center|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_order.rst
    :param order [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_order.rst
    :param PVAJ[REAL(C_DOUBLE), dimension(1\:12), intent(out)]: .. include:: arg_PVAJ_order.rst
    :r calceph_compute_order: |retfuncfails0|
    :rtype calceph_compute_order: INTEGER(C_INT)


.. include:: calceph.multiple_compute_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_orient_order (eph, JD0, time, target, unit, order, PVAJ ) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_order.rst
    :param order [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_order_orient.rst
    :param PVAJ[REAL(C_DOUBLE), dimension(1\:12), intent(out)]: .. include:: arg_PVAJ_orient_order.rst
    :r calceph_compute_order: |retfuncfails0|
    :rtype calceph_compute_order: INTEGER(C_INT)


.. include:: calceph.multiple_orient_order.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_rotangmom_order (eph, JD0, time, target, unit, order, PVAJ ) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param JD0 [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_JD0|
    :param time [REAL(C_DOUBLE), VALUE, intent(in)]: |arg_time|
    :param target [INTEGER(C_INT), VALUE, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_order.rst
    :param order [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_order_rotangmom.rst
    :param PVAJ[REAL(C_DOUBLE), dimension(1\:12), intent(out)]: .. include:: arg_PVAJ_rotangmom_order.rst
    :r calceph_rotangmom_order: |retfuncfails0|
    :rtype calceph_rotangmom_order: INTEGER(C_INT)


.. include:: calceph.multiple_rotangmom_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getconstant (eph, name, value) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_constant_name|.
    :p  value [REAL(C_DOUBLE), intent(out)]: |arg_constant_value|.
    :r calceph_getconstant: |retfuncfailsnbval|
    :rtype calceph_getconstant: INTEGER(C_INT)


.. include:: calceph.multiple_getconstant.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstantsd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getconstantsd (eph, name, value) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_constant_name|.
    :p  value [REAL(C_DOUBLE), intent(out)]: |arg_constant_value|.
    :r calceph_getconstantsd: |retfuncfailsnbval|
    :rtype calceph_getconstantsd: INTEGER(C_INT)

.. include:: calceph.multiple_getconstantsd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getconstantvd (eph, name, arrayvalue, nvalue) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_constant_name|.
    :p  value [REAL(C_DOUBLE), dimension(1\:nvalue), intent(out)]: |arg_constant_arrayvalue|.
    :p  nvalue [INTEGER(C_INT), VALUE, intent(in)]: |arg_constant_nvalue|
    :r calceph_getconstantvd: |retfuncfailsnbval|
    :rtype calceph_getconstantvd: INTEGER(C_INT)

This function stores, to the array *arrayvalue* as floating-point numbers, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value..

The required value *nvalue* to store all values can be determinated with the previous call to *calceph_getconstantsd*.

.. include:: calceph.multiple_getconstantvd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getconstantss (eph, name, value) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_constant_name|.
    :p  value [CHARACTER(len=1,kind=C_CHAR), dimension(CALCEPH_MAX_CONSTANVALUE), intent(out)]: |arg_constant_value|.
    :r calceph_getconstantss: |retfuncfailsnbval|
    :rtype calceph_getconstantss: INTEGER(C_INT)



.. include:: calceph.multiple_getconstantss.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvs|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getconstantvs (eph, name, arrayvalue, nvalue) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|.
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_constant_name|.
    :p  value [CHARACTER(len=1,kind=C_CHAR), dimension(1\:nvalue), intent(out)]: |arg_constant_arrayvalue|.
    :p  nvalue [INTEGER(C_INT), VALUE, intent(in)]: |arg_constant_nvalue|
    :r calceph_getconstantvs: |retfuncfailsnbval|
    :rtype calceph_getconstantvs: INTEGER(C_INT)

This function stores, to the array *arrayvalue* as strings of characters, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value.

The required value *nvalue* to store all values can be determinated with the previous call to *calceph_getconstantss*.

.. include:: calceph.multiple_getconstantvs.rst


.. %------------------------------------------------

|menu_calceph_getconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getconstantcount (eph) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :r calceph_getconstantcount: |arg_constant_number|. |retfuncfails0|
    :rtype calceph_getconstantcount: INTEGER(C_INT)


.. include:: calceph.multiple_getconstantcount.rst

.. %------------------------------------------------

|menu_calceph_getconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getconstantindex (eph, index, name, value) BIND(C)

    :p  eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  index [INTEGER(C_INT), VALUE, intent(in)]: |arg_constant_index|
    :p  name [CHARACTER(len=1,kind=C_CHAR),  dimension(CALCEPH_MAX_CONSTANTNAME), intent(out)]: |arg_constant_name|.
    :p  value [REAL(C_DOUBLE), intent(out)]: |arg_constant_value|
    :r calceph_getconstantindex: |retfuncfailsnbval|
    :rtype calceph_getconstantindex: INTEGER(C_INT)

.. include:: calceph.multiple_getconstantindex.rst

.. %------------------------------------------------

|menu_calceph_getfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getfileversion (eph, version) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :param  version [CHARACTER(len=1,kind=C_CHAR), dimension(CALCEPH_MAX_CONSTANTVALUE), intent(out)]: |arg_fileversion|
    :r calceph_getfileversion: |retfuncnotfound0|
    :rtype calceph_getfileversion: INTEGER(C_INT)

.. include:: calceph.multiple_getfileversion.rst

.. %------------------------------------------------

|menu_calceph_getidbyname|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getidbyname (eph, name, unit, id) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  name [CHARACTER(len=1,kind=C_CHAR), intent(in)]: |arg_name_getidbyname|.
    :p  unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_idbyname.rst
    :p  id [INTEGER(C_INT), intent(out)]: |arg_id_getidbyname|
    :r calceph_getidbyname: |retfuncnotfoundid0|
    :rtype calceph_getidbyname: INTEGER(C_INT)

This function returns, in the parameter *id*,  the identification number of the body associated to the given name.

.. include:: calceph.multiple_getidbyname.rst


.. %------------------------------------------------

|menu_calceph_getnamebyidss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getnamebyidss (eph, id, unit, name) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  id [INTEGER(C_INT), intent(out)]: |arg_id_getidbyname|
    :p  unit [INTEGER(C_INT), VALUE, intent(in)]: .. include:: arg_unit_idbyname.rst
    :p  name [CHARACTER(len=1,kind=C_CHAR), dimension(CALCEPH_MAX_CONSTANTVALUE), intent(out)]: |arg_name_getidbyname|
    :r calceph_getnamebyidss: |retfuncnotfoundid0|
    :rtype calceph_getnamebyidss: INTEGER(C_INT)

This function returns, in the parameter *name*, the first given name of the body associated to the identification number *id*.

To remove the C null character at the end of the name, you could use the following statement :  name = name(1:index(name, C_NULL_CHAR)-1)
 

.. include:: calceph.multiple_getnamebyidss.rst


.. %------------------------------------------------

|menu_calceph_gettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_gettimescale (eph) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :r calceph_gettimescale: |retfuncfails0|
    :rtype calceph_gettimescale: INTEGER(C_INT)


.. include:: calceph.multiple_gettimescale.rst

.. %------------------------------------------------

|menu_calceph_gettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_gettimespan (eph, firsttime, lasttime, continuous) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  firsttime [REAL(C_DOUBLE), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(C_DOUBLE), intent(out)]: |arg_lasttime|
    :p  continuous [INTEGER(C_INT), intent(out)]: |arg_continuous|
    :r calceph_gettimespan: |retfuncfails0|
    :rtype calceph_gettimespan: INTEGER(C_INT)

.. include:: calceph.multiple_gettimespan.rst


.. %------------------------------------------------

|menu_calceph_getpositionrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getpositionrecordcount (eph) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :r calceph_getpositionrecordcount: |arg_positionrecordcount|. |retfuncfails0|
    :rtype calceph_getpositionrecordcount: INTEGER(C_INT)

.. include:: calceph.multiple_getpositionrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getpositionrecordindex (eph, index, target, center, firsttime, lasttime, frame) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  index [INTEGER(C_INT), VALUE, intent(int)]: |arg_positionrecord_index|
    :p  target [INTEGER(C_INT), intent(out)]: |arg_positionrecord_target|
    :p  center [INTEGER(C_INT), intent(out)]: |arg_positionrecord_center|
    :p  firsttime [REAL(C_DOUBLE), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(C_DOUBLE), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER(C_INT), intent(out)]: |arg_frame|
    :r calceph_getpositionrecordindex: |retfuncfails0|
    :rtype calceph_getpositionrecordindex: INTEGER(C_INT)


.. include:: calceph.multiple_getpositionrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getpositionrecordindex2 (eph, index, target, center, firsttime, lasttime, frame, segid) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  index [INTEGER(C_INT), VALUE, intent(int)]: |arg_positionrecord_index|
    :p  target [INTEGER(C_INT), intent(out)]: |arg_positionrecord_target|
    :p  center [INTEGER(C_INT), intent(out)]: |arg_positionrecord_center|
    :p  firsttime [REAL(C_DOUBLE), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(C_DOUBLE), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER(C_INT), intent(out)]: |arg_frame|
    :p  segid [INTEGER(C_INT), intent(out)]: |arg_segid|
    :r calceph_getpositionrecordindex2: |retfuncfails0|
    :rtype calceph_getpositionrecordindex2: INTEGER(C_INT)


.. include:: calceph.multiple_getpositionrecordindex2.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getorientrecordcount (eph) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :r calceph_getorientrecordcount: |arg_orientrecordcount|. |retfuncfails0|
    :rtype calceph_getorientrecordcount: INTEGER(C_INT)


.. include:: calceph.multiple_getorientrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: calceph_getorientrecordindex (eph, index, target, firsttime, lasttime, frame) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  index [INTEGER(C_INT), VALUE, intent(int)]: |arg_orientrecord_index|
    :p  target [INTEGER(C_INT), intent(out)]: |arg_positionrecord_target|
    :p  firsttime [REAL(C_DOUBLE), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(C_DOUBLE), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER(C_INT), intent(out)]: |arg_frame|
    :r calceph_getorientrecordindex: |retfuncfails0|
    :rtype calceph_getorientrecordindex: INTEGER(C_INT)


.. include:: calceph.multiple_getorientrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: calceph_getorientrecordindex2 (eph, index, target, firsttime, lasttime, frame, segid) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p  index [INTEGER(C_INT), VALUE, intent(int)]: |arg_orientrecord_index|
    :p  target [INTEGER(C_INT), intent(out)]: |arg_positionrecord_target|
    :p  firsttime [REAL(C_DOUBLE), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(C_DOUBLE), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER(C_INT), intent(out)]: |arg_frame|
    :p  segid [INTEGER(C_INT), intent(out)]: |arg_segid|
    :r calceph_getorientrecordindex2: |retfuncfails0|
    :rtype calceph_getorientrecordindex2: INTEGER(C_INT)

.. include:: calceph.multiple_getorientrecordindex2.rst

.. %----------------------------------------------------------------------------

|menu_calceph_close|
~~~~~~~~~~~~~~~~~~~~


.. f:subroutine:: calceph_close (eph) BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|


This function closes the access associated to |ephemerisdescriptoreph| and frees allocated memory for it.



