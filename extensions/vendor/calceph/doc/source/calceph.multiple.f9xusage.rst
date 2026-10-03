.. _`Reading/Evaluation functions`:

.. include:: calceph.multiple_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_open|
~~~~~~~~~~~~~~~~~~~

.. f:function:: function f90calceph_open (eph, filename)

    :p  filename [CHARACTER(len=*), intent(in)]: |arg_filename|.
    :p  eph   [INTEGER(8), intent(out)]: |arg_eph|
    :r f90calceph_open: |retfuncfails0|
    :rtype f90calceph_open: INTEGER

.. include:: calceph.multiple_open.rst

.. %----------------------------------------------------------------------------

|menu_calceph_open_array|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_open_array (eph, n, array_filename, len_filename)

    :p  eph   [INTEGER(8), intent(out)]: |arg_eph|
    :p  n [INTEGER, intent(in)]: |arg_n|.
    :p  array_filename [CHARACTER(len=*), dimension(*), intent(in)]: |arg_array_filename|
    :p  len_filename [INTEGER, intent(in)]: |arg_len_filename|.
    :r f90calceph_open_array: |retfuncfails0|
    :rtype f90calceph_open_array: INTEGER


.. include:: calceph.multiple_open_array.rst

.. %----------------------------------------------------------------------------

|menu_calceph_prefetch|
~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_prefetch (eph)

    :p  eph   [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_prefetch: |retfuncfails0|
    :rtype f90calceph_prefetch: INTEGER

.. include:: calceph.multiple_prefetch.rst

.. %----------------------------------------------------------------------------

|menu_calceph_isthreadsafe|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_isthreadsafe (eph)

    :p  eph   [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_isthreadsafe: returns 1 if multiple threads can access the same ephemeris ephemeris descriptor, otherwise 0.
    :rtype f90calceph_isthreadsafe: INTEGER


.. include:: calceph.multiple_isthreadsafe.rst


.. %----------------------------------------------------------------------------

|menu_calceph_compute|
~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_compute (eph, JD0, time, target, center, PV )

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target|
    :param center [INTEGER, intent(in)]: |arg_center|
    :param PV(6) [REAL(8), intent(out)]: .. include:: arg_PV.rst
    :r f90calceph_compute: |retfuncfails0|
    :rtype f90calceph_compute: INTEGER


.. include:: calceph.multiple_compute.rst

   
.. %----------------------------------------------------------------------------

|menu_calceph_compute_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_compute_unit (eph, JD0, time, target, center, unit, PV )

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_unit|
    :param center [INTEGER, intent(in)]: |arg_center_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit.rst
    :param PV(6) [REAL(8), intent(out)]: .. include:: arg_PV_unit.rst
    :r f90calceph_compute_unit: |retfuncfails0|
    :rtype f90calceph_compute_unit: INTEGER

.. include:: calceph.multiple_compute_unit.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_orient_unit (eph, JD0, time, target, unit, PV)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param PV(6) [REAL(8), intent(out)]: .. include:: arg_PV_orient_unit.rst
    :r f90calceph_orient_unit: |retfuncfails0|
    :rtype f90calceph_orient_unit: INTEGER


.. include:: calceph.multiple_orient_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_rotangmom_unit (eph, JD0, time, target, unit, PV)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param PV(6) [REAL(8), intent(out)]: .. include:: arg_PV_rotangmom_unit.rst
    :r f90calceph_rotangmom_unit: |retfuncfails0|
    :rtype f90calceph_rotangmom_unit: INTEGER



.. include:: calceph.multiple_rotangmom_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_compute_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_compute_order (eph, JD0, time, target, center, unit, order, PVAJ )

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_unit|
    :param center [INTEGER, intent(in)]: |arg_center_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit_order.rst
    :param order [INTEGER, intent(in)]: .. include:: arg_order.rst
    :param PVAJ(12) [REAL(8), intent(out)]: .. include:: arg_PVAJ_order.rst
    :r f90calceph_compute_order: |retfuncfails0|
    :rtype f90calceph_compute_order: INTEGER


.. include:: calceph.multiple_compute_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_orient_order (eph, JD0, time, target,  unit, order, PVAJ )

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param order [INTEGER, intent(in)]: .. include:: arg_order_orient.rst
    :param PVAJ(12) [REAL(8), intent(out)]: .. include:: arg_PVAJ_orient_order.rst
    :r f90calceph_orient_order: |retfuncfails0|
    :rtype f90calceph_orient_order: INTEGER


.. include:: calceph.multiple_orient_order.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_rotangmom_order (eph, JD0, time, target,  unit, order, PVAJ )

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param JD0 [REAL(8), intent(in)]: |arg_JD0|
    :param time [REAL(8), intent(in)]: |arg_time|
    :param target [INTEGER, intent(in)]: |arg_target_orient_unit|
    :param unit [INTEGER, intent(in)]: .. include:: arg_unit_orient_unit.rst
    :param order [INTEGER, intent(in)]: .. include:: arg_order_rotangmom.rst
    :param PVAJ(12) [REAL(8), intent(out)]: .. include:: arg_PVAJ_rotangmom_order.rst
    :r f90calceph_rotangmom_order: |retfuncfails0|
    :rtype f90calceph_rotangmom_order: INTEGER


.. include:: calceph.multiple_rotangmom_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getconstant (eph, name, value) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_constant_name|
    :p  value [REAL(8), intent(out)]: |arg_constant_value|
    :r f90calceph_getconstant: |retfuncfailsnbval|
    :rtype f90calceph_getconstant: INTEGER


.. include:: calceph.multiple_getconstant.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstantsd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getconstantsd (eph, name, value) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_constant_name|
    :p  value [REAL(8), intent(out)]: |arg_constant_value|
    :r f90calceph_getconstantsd: |retfuncfailsnbval|
    :rtype f90calceph_getconstantsd: INTEGER



.. include:: calceph.multiple_getconstantsd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_getconstantvd (eph, name, arrayvalue, nvalue) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_constant_name|
    :p  value [REAL(8), dimension(1\:nvalue), intent(inout)]: |arg_constant_arrayvalue|
    :p  nvalue [INTEGER, intent(in)]: |arg_constant_nvalue|
    :r f90calceph_getconstantvd: |retfuncfailsnbval|
    :rtype f90calceph_getconstantvd: INTEGER
    
This function stores, to the array *arrayvalue* as floating-point numbers, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value..

The required value *nvalue* to store all values can be determinated with the previous call to *f90calceph_getconstantsd*.



.. include:: calceph.multiple_getconstantvd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getconstantss (eph, name, value) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_constant_name|
    :p  value [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), intent(out)]: |arg_constant_value|
    :r f90calceph_getconstantss: |retfuncfailsnbval|
    :rtype f90calceph_getconstantss: INTEGER


.. include:: calceph.multiple_getconstantss.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvs|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_getconstantvs (eph, name, arrayvalue, nvalue) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_constant_name|
    :p  value [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), dimension(1\:nvalue), intent(inout)]: |arg_constant_arrayvalue|
    :p  nvalue [INTEGER, intent(in)]: |arg_constant_nvalue|
    :r f90calceph_getconstantvs: |retfuncfailsnbval|
    :rtype f90calceph_getconstantvs: INTEGER
    
This function stores, to the array *arrayvalue* as strings of characters, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value.

The required value *nvalue* to store all values can be determinated with the previous call to *f90calceph_getconstantss*.

.. include:: calceph.multiple_getconstantvs.rst


.. %------------------------------------------------

|menu_calceph_getconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getconstantcount (eph) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_getconstantcount: |arg_constant_number|. |retfuncfails0|
    :rtype f90calceph_getconstantcount: INTEGER


.. include:: calceph.multiple_getconstantcount.rst

.. %------------------------------------------------

|menu_calceph_getconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_getconstantindex (eph, index, name, value) 

    :p  eph  [INTEGER(8), intent(in)]: |arg_eph|
    :p  index [INTEGER, intent(in)]: |arg_constant_index|
    :p  name  [CHARACTER(len=CALCEPH_MAX_CONSTANTNAME), intent(out)]: |arg_constant_name|
    :p  value [REAL(8), intent(out)]: |arg_constant_value|
    :r f90calceph_getconstantindex: |retfuncfailsnbval|
    :rtype f90calceph_getconstantindex: INTEGER


.. include:: calceph.multiple_getconstantindex.rst

.. %------------------------------------------------

|menu_calceph_getfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_getfileversion (eph, version) 
    
    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :param  version [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), intent(out)]: |arg_fileversion|
    :r f90calceph_getfileversion: |retfuncnotfound0|
    :rtype f90calceph_getfileversion: INTEGER



.. include:: calceph.multiple_getfileversion.rst

.. %------------------------------------------------

|menu_calceph_getidbyname|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getidbyname (eph, name, unit, id) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  name  [CHARACTER(len=*), intent(in)]: |arg_name_getidbyname|
    :p  unit [INTEGER, intent(in)]: .. include:: arg_unit_idbyname.rst
    :p  id [INTEGER, intent(out)]: |arg_id_getidbyname|
    :r f90calceph_getidbyname: |retfuncnotfoundid0|
    :rtype f90calceph_getidbyname: INTEGER

This function returns, in the parameter *id*, the identification number of the body associated to the given name.

.. include:: calceph.multiple_getidbyname.rst


.. %------------------------------------------------

|menu_calceph_getnamebyidss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getnamebyidss (eph, id, unit, name) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  id [INTEGER, intent(out)]: |arg_id_getidbyname|
    :p  unit [INTEGER, intent(in)]: .. include:: arg_unit_idbyname.rst
    :param name [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), intent(out)]: |arg_name_getidbyname|
    :r f90calceph_getnamebyidss: |retfuncnotfoundid0|
    :rtype f90calceph_getnamebyidss: INTEGER

This function returns, in the parameter *name*, the first given name of the body associated to the identification number *id*.

.. include:: calceph.multiple_getnamebyidss.rst


.. %------------------------------------------------

|menu_calceph_gettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_gettimescale (eph) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_gettimescale: |retfuncfails0|
    :rtype f90calceph_gettimescale: INTEGER

.. include:: calceph.multiple_gettimescale.rst

.. %------------------------------------------------

|menu_calceph_gettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_gettimespan (eph, firsttime, lasttime, continuous) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  firsttime [REAL(8), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(8), intent(out)]: |arg_lasttime|
    :p  continuous [INTEGER, intent(out)]: |arg_continuous|
    :r f90calceph_gettimespan: |retfuncfails0|
    :rtype f90calceph_gettimespan: INTEGER


.. include:: calceph.multiple_gettimespan.rst


.. %------------------------------------------------

|menu_calceph_getpositionrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getpositionrecordcount (eph) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_getpositionrecordcount: |arg_positionrecordcount|. |retfuncfails0|
    :rtype f90calceph_getpositionrecordcount: INTEGER


.. include:: calceph.multiple_getpositionrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getpositionrecordindex (eph, index, target, center, firsttime, lasttime, frame) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  index [INTEGER, intent(int)]: |arg_positionrecord_index|
    :p  target [INTEGER, intent(out)]: |arg_positionrecord_target|
    :p  center [INTEGER, intent(out)]: |arg_positionrecord_center|
    :p  firsttime [REAL(8), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(8), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER, intent(out)]: |arg_frame|
    :r f90calceph_getpositionrecordindex: |retfuncfails0|
    :rtype f90calceph_getpositionrecordindex: INTEGER



.. include:: calceph.multiple_getpositionrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. f:function:: f90calceph_getpositionrecordindex2 (eph, index, target, center, firsttime, lasttime, frame, segid) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  index [INTEGER, intent(int)]: |arg_positionrecord_index|
    :p  target [INTEGER, intent(out)]: |arg_positionrecord_target|
    :p  center [INTEGER, intent(out)]: |arg_positionrecord_center|
    :p  firsttime [REAL(8), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(8), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER, intent(out)]: |arg_frame|
    :p  segid [INTEGER, intent(out)]: |arg_segid|
    :r f90calceph_getpositionrecordindex2: |retfuncfails0|
    :rtype f90calceph_getpositionrecordindex2: INTEGER



.. include:: calceph.multiple_getpositionrecordindex2.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getorientrecordcount (eph) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :r f90calceph_getorientrecordcount: |arg_orientrecordcount|. |retfuncfails0|
    :rtype f90calceph_getorientrecordcount: INTEGER


.. include:: calceph.multiple_getorientrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getpositionrecordindex (eph, index, target, firsttime, lasttime, frame) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  index [INTEGER, intent(int)]: |arg_orientrecord_index|
    :p  target [INTEGER, intent(out)]: |arg_positionrecord_target|
    :p  firsttime [REAL(8), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(8), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER, intent(out)]: |arg_frame|
    :r f90calceph_getorientrecordindex: |retfuncfails0|
    :rtype f90calceph_getorientrecordindex: INTEGER



.. include:: calceph.multiple_getorientrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. f:function:: f90calceph_getpositionrecordindex2 (eph, index, target, firsttime, lasttime, frame, segid) 

    :param eph [INTEGER(8), intent(in)]: |arg_eph|
    :p  index [INTEGER, intent(int)]: |arg_orientrecord_index|
    :p  target [INTEGER, intent(out)]: |arg_positionrecord_target|
    :p  firsttime [REAL(8), intent(out)]: |arg_firsttime|
    :p  lasttime [REAL(8), intent(out)]: |arg_lasttime|
    :p  frame [INTEGER, intent(out)]: |arg_frame|
    :p  segid [INTEGER, intent(out)]: |arg_segid|
    :r f90calceph_getorientrecordindex2: |retfuncfails0|
    :rtype f90calceph_getorientrecordindex2: INTEGER


.. include:: calceph.multiple_getorientrecordindex2.rst

.. %----------------------------------------------------------------------------

|menu_calceph_close|
~~~~~~~~~~~~~~~~~~~~

.. f:subroutine:: f90calceph_close (eph)

    :param eph [INTEGER(8), intent(in)]: |arg_eph|


This function closes the access associated to |ephemerisdescriptoreph| and frees allocated memory for it.



