.. _`Reading/Evaluation functions`:

.. include:: calceph.multiple_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_open|
~~~~~~~~~~~~~~~~~~~

.. c:function:: t_calcephbin* calceph_open ( const char *filename )

    :param filename: |arg_filename|
    :return: |arg_eph|. |retfuncfailsNULL|

.. include:: calceph.multiple_open.rst

.. %----------------------------------------------------------------------------

|menu_calceph_open_array|
~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: t_calcephbin* calceph_open_array (int n, const char *array_filename[] )

    :param  n: |arg_n|
    :param  array_filename: |arg_array_filename|
    :return: |arg_eph|. |retfuncfailsNULL|


.. include:: calceph.multiple_open_array.rst

.. %----------------------------------------------------------------------------

|menu_calceph_prefetch|
~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_prefetch ( t_calcephbin* eph )

    :param  eph: |arg_eph|
    :return: |retfuncfails0|

.. include:: calceph.multiple_prefetch.rst

.. %----------------------------------------------------------------------------

|menu_calceph_isthreadsafe|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_isthreadsafe ( t_calcephbin* eph )

    :param  eph: |arg_eph|
    :return: returns 1 if multiple threads can access the same ephemeris ephemeris descriptor, otherwise 0.


.. include:: calceph.multiple_isthreadsafe.rst


.. %----------------------------------------------------------------------------

|menu_calceph_compute|
~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_compute (t_calcephbin* eph, double JD0, double time, int target, int center, double PV[6] )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target|
    :param  center: |arg_center|
    :param  PV:  .. include:: arg_PV.rst
    :return: |retfuncfails0|


.. include:: calceph.multiple_compute.rst

   
.. %----------------------------------------------------------------------------

|menu_calceph_compute_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_compute_unit (t_calcephbin* eph, double JD0, double time, int target, int center, int unit, double PV[6] )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_unit|
    :param  center: |arg_center_unit|
    :param  unit: .. include:: arg_unit.rst
    :param  PV:  .. include:: arg_PV_unit.rst
    :return: |retfuncfails0|

.. include:: calceph.multiple_compute_unit.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_orient_unit (t_calcephbin* eph, double JD0, double time, int target, int unit, double PV[6] )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_orient_unit|
    :param  unit: .. include:: arg_unit_orient_unit.rst
    :param  PV:  .. include:: arg_PV_orient_unit.rst
    :return: |retfuncfails0|


.. include:: calceph.multiple_orient_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_rotangmom_unit (t_calcephbin* eph, double JD0, double time, int target, int unit, double PV[6] )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_orient_unit|
    :param  unit: .. include:: arg_unit_orient_unit.rst
    :param  PV:  .. include:: arg_PV_rotangmom_unit.rst
    :return: |retfuncfails0|


.. include:: calceph.multiple_rotangmom_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_compute_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_compute_order (t_calcephbin* eph, double JD0, double time, int target, int center, int unit, int order, double *PVAJ )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_unit|
    :param  center: |arg_center_unit|
    :param  unit: .. include:: arg_unit_order.rst
    :param  order: .. include:: arg_order.rst
    :param  PVAJ:  .. include:: arg_PVAJ_order.rst
    :return: |retfuncfails0|

.. include:: calceph.multiple_compute_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_orient_order (t_calcephbin* eph, double JD0, double time, int target, int unit, int order, double *PVAJ )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_orient_unit|
    :param  unit: .. include:: arg_unit_orient_unit.rst
    :param  order: .. include:: arg_order_orient.rst
    :param  PVAJ:  .. include:: arg_PVAJ_orient_order.rst
    :return: |retfuncfails0|


.. include:: calceph.multiple_orient_order.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_rotangmom_order (t_calcephbin* eph, double JD0, double time, int target, int unit, int order, double *PVAJ )

    :param  eph: |arg_eph|
    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target_orient_unit|
    :param  unit: .. include:: arg_unit_orient_unit.rst
    :param  order: .. include:: arg_order_rotangmom.rst
    :param  PVAJ:  .. include:: arg_PVAJ_rotangmom_order.rst
    :return: |retfuncfails0|


.. include:: calceph.multiple_rotangmom_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_getconstant (t_calcephbin*  eph, const char* name, double *value )

    :param  eph: |arg_eph|
    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfailsnbval|

.. include:: calceph.multiple_getconstant.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstantsd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantsd (t_calcephbin*  eph, const char* name, double *value )

    :param  eph: |arg_eph|
    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfailsnbval|


.. include:: calceph.multiple_getconstantsd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantvd (t_calcephbin*  eph, const char* name, double *arrayvalue, int nvalue)

    :param  eph: |arg_eph|
    :param  name: |arg_constant_name|
    :param  arrayvalue: |arg_constant_arrayvalue|
    :param  nvalue: |arg_constant_nvalue|
    :return: |retfuncfailsnbval|

This function stores, to the array *arrayvalue* as floating-point numbers, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value..

The required value *nvalue* to store all values can be determinated with this previous call *calceph_getconstantvd(eph, name, NULL, 0)*.

.. include:: calceph.multiple_getconstantvd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantss (t_calcephbin*  eph, const char* name, t_calcephcharvalue value )

    :param  eph: |arg_eph|
    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfailsnbval|


.. include:: calceph.multiple_getconstantss.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvs|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantvs (t_calcephbin*  eph, const char* name, t_calcephcharvalue *arrayvalue, int nvalue)

    :param  eph: |arg_eph|
    :param  name: |arg_constant_name|
    :param  arrayvalue: |arg_constant_arrayvalue|
    :param  nvalue: |arg_constant_nvalue|
    :return: |retfuncfailsnbval|

This function stores, to the array *arrayvalue* as strings of characters, the *nvalue* first values associated to the constant *name* in the header of the ephemeris file |eph|. The integer value returned by the function is equal to the number of valid entries in the *arrayvalue* if *nvalue* is greater or equal to that integer value.

The required value *nvalue* to store all values can be determinated with this previous call *calceph_getconstantvs(eph, name, NULL, 0)*.

.. include:: calceph.multiple_getconstantvs.rst


.. %------------------------------------------------

|menu_calceph_getconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantcount (t_calcephbin* eph )

    :param  eph: |arg_eph|
    :return: |arg_constant_number|. |retfuncfails0|

.. include:: calceph.multiple_getconstantcount.rst

.. %------------------------------------------------

|menu_calceph_getconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getconstantindex (t_calcephbin* eph, int index, char name[CALCEPH_MAX_CONSTANTNAME], double *value)

    :param  eph: |arg_eph|
    :param  index: |arg_constant_index|
    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfailsnbval|


.. include:: calceph.multiple_getconstantindex.rst

.. %------------------------------------------------

|menu_calceph_getfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_getfileversion (t_calcephbin* eph, char version[CALCEPH_MAX_CONSTANTVALUE])

    :param  eph: |arg_eph|
    :param  version: |arg_fileversion|
    :return: |retfuncnotfound0|

.. include:: calceph.multiple_getfileversion.rst

.. %------------------------------------------------

|menu_calceph_getidbyname|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getidbyname (t_calcephbin* eph, const char *name, int unit, int* id )

    :param  eph: |arg_eph|
    :param  name: |arg_name_getidbyname|
    :param  unit: .. include:: arg_unit_idbyname.rst
    :param  id: |arg_id_getidbyname|
    :return: |retfuncnotfoundid0|

This function returns, in the parameter *id*, the identification number of the body associated to the given name.

.. include:: calceph.multiple_getidbyname.rst


.. %------------------------------------------------

|menu_calceph_getnamebyidss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getnamebyidss (t_calcephbin* eph, int id , int unit, char name[CALCEPH_MAX_CONSTANTVALUE])

    :param  eph: |arg_eph|
    :param  name: |arg_name_getidbyname|
    :param  unit: .. include:: arg_unit_idbyname.rst
    :param  id: |arg_id_getidbyname|
    :return: |retfuncnotfoundid0|

This function returns, in the parameter *name*, the first given name of the body associated to the identification number *id*.

.. include:: calceph.multiple_getnamebyidss.rst


.. %------------------------------------------------

|menu_calceph_gettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_gettimescale (t_calcephbin* eph)

    :param  eph: |arg_eph|
    :return: |retfuncfails0|

.. include:: calceph.multiple_gettimescale.rst

.. %------------------------------------------------

|menu_calceph_gettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_gettimespan (t_calcephbin* eph, double* firsttime, double* lasttime, int* continuous )

    :param  eph: |arg_eph|
    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  continuous: |arg_continuous|
    :return: |retfuncfails0|

.. include:: calceph.multiple_gettimespan.rst


.. %------------------------------------------------

|menu_calceph_getpositionrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getpositionrecordcount (t_calcephbin* eph)

    :param  eph: |arg_eph|
    :return: |arg_positionrecordcount|. |retfuncfails0|

.. include:: calceph.multiple_getpositionrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getpositionrecordindex (t_calcephbin* eph, int index, int* target, int* center, double* firsttime, double* lasttime, int* frame)

    :param  eph: |arg_eph|
    :param  index: |arg_positionrecord_index|
    :param  target: |arg_positionrecord_target|
    :param  center: |arg_positionrecord_center|
    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  frame: |arg_frame|
    :return: |retfuncfails0|


.. include:: calceph.multiple_getpositionrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getpositionrecordindex2 (t_calcephbin* eph, int index, int* target, int* center, double* firsttime, double* lasttime, int* frame, int *segid)

    :param  eph: |arg_eph|
    :param  index: |arg_positionrecord_index|
    :param  target: |arg_positionrecord_target|
    :param  center: |arg_positionrecord_center|
    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  frame: |arg_frame|
    :param  segid: |arg_segid|
    :return: |retfuncfails0|

.. include:: calceph.multiple_getpositionrecordindex2.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_getorientrecordcount (t_calcephbin* eph)

    :param  eph: |arg_eph|
    :return: |arg_orientrecordcount|. |retfuncfails0|

.. include:: calceph.multiple_getorientrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getorientrecordindex (t_calcephbin* eph, int index, int* target, double* firsttime, double* lasttime, int* frame)

    :param  eph: |arg_eph|
    :param  index: |arg_orientrecord_index|
    :param  target: |arg_positionrecord_target|
    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  frame: |arg_frame|
    :return: |retfuncfails0|

.. include:: calceph.multiple_getorientrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_getorientrecordindex2 (t_calcephbin* eph, int index, int* target, double* firsttime, double* lasttime, int* frame, int *segid)

    :param  eph: |arg_eph|
    :param  index: |arg_orientrecord_index|
    :param  target: |arg_positionrecord_target|
    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  frame: |arg_frame|
    :param  segid: |arg_segid|
    :return: |retfuncfails0|


.. include:: calceph.multiple_getorientrecordindex2.rst

.. %----------------------------------------------------------------------------

|menu_calceph_close|
~~~~~~~~~~~~~~~~~~~~


.. c:function:: void calceph_close (t_calcephbin* eph )

    :param  eph: |arg_eph|


This function closes the access associated to |ephemerisdescriptoreph| and frees allocated memory for it.



