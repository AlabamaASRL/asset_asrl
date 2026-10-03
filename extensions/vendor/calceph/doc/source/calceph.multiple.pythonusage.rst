.. _`Reading/Evaluation functions`:

.. include:: calceph.multiple_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_open|
~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.open (filename) -> eph

    :param  str filename: |arg_filename|
    :return: |arg_eph|
    :rtype: calcephpy.CalcephBin 


.. include:: calceph.multiple_open.rst

.. %----------------------------------------------------------------------------

.. % |menu_calceph_open_array|
.. % ~~~~~~~~~~~~~~~~~~~~~~~~~

:noindex:

.. py:function:: calcephpy.CalcephBin.open ( array_filename ) -> eph

    :param  list array_filename: |arg_array_filename|
    :return: |arg_eph|
    :rtype: calcephpy.CalcephBin 


.. include:: calceph.multiple_open_array.rst

.. %----------------------------------------------------------------------------

|menu_calceph_prefetch|
~~~~~~~~~~~~~~~~~~~~~~~


.. py:method:: calcephpy.CalcephBin.prefetch()


.. include:: calceph.multiple_prefetch.rst

.. %----------------------------------------------------------------------------

|menu_calceph_isthreadsafe|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:method:: calcephpy.CalcephBin.isthreadsafe()


.. include:: calceph.multiple_isthreadsafe.rst


.. %----------------------------------------------------------------------------

|menu_calceph_compute|
~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.compute (JD0, time, target, center) -> PV

    :param  float/list/numpy.ndarray JD0: |arg_JD0|
    :param  float/list/numpy.ndarray time: |arg_time|
    :param  int target: |arg_target|
    :param  int center: |arg_center|
    :return:  .. include:: arg_PV.rst
    :rtype: list

.. include:: calceph.multiple_compute.rst

   
.. %----------------------------------------------------------------------------

|menu_calceph_compute_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.compute_unit (JD0, time, target, center, unit) -> PV

    :param  float/list/numpy.ndarray JD0: |arg_JD0|
    :param  float/list/numpy.ndarray time: |arg_time|
    :param  int target: |arg_target_unit|
    :param  int center: |arg_center_unit|
    :param  int unit: .. include:: arg_unit.rst
    :return:  .. include:: arg_PV_unit.rst
    :rtype: list

.. include:: calceph.multiple_compute_unit.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.orient_unit (JD0, time, target,  unit) -> PV

    :param  float/list/numpy.ndarray JD0: |arg_JD0|
    :param  float/list/numpy.ndarray time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :return:  .. include:: arg_PV_orient_unit.rst
    :rtype: list

.. include:: calceph.multiple_orient_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.rotangmom_unit (JD0, time, target, unit) -> PV

    :param  float JD0: |arg_JD0|
    :param  float time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :return:  .. include:: arg_PV_rotangmom_unit.rst
    :rtype: list

.. include:: calceph.multiple_rotangmom_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_compute_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.compute_order(JD0, time, target, center, unit, order) -> PVAJ

    :param  float/list/numpy.ndarray JD0: |arg_JD0|
    :param  float/list/numpy.ndarray time: |arg_time|
    :param  int target: |arg_target_unit|
    :param  int center: |arg_center_unit|
    :param  int unit: .. include:: arg_unit_order.rst
    :param  int order: .. include:: arg_order.rst
    :return:  .. include:: arg_PVAJ_order.rst
    :rtype: list


.. include:: calceph.multiple_compute_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.orient_order(JD0, time, target, unit, order) -> PVAJ

    :param  float/list/numpy.ndarray JD0: |arg_JD0|
    :param  float/list/numpy.ndarray time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :param  int order: .. include:: arg_order_orient.rst
    :return: .. include:: arg_PVAJ_orient_order.rst
    :rtype: list

.. include:: calceph.multiple_orient_order.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.rotangmom_order(JD0, time, target, unit, order) -> PVAJ

    :param  float JD0: |arg_JD0|
    :param  float time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :param  int order: .. include:: arg_order_rotangmom.rst
    :return: .. include:: arg_PVAJ_rotangmom_order.rst
    :rtype: list


.. include:: calceph.multiple_rotangmom_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getconstant(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: float


.. include:: calceph.multiple_getconstant.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstantsd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getconstantsd(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: float


.. include:: calceph.multiple_getconstantsd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getconstantvd(name) -> arrayvalue

    :param  str name: |arg_constant_name|
    :return: |arg_constant_arrayvalue|
    :rtype: list

This function returns, as floating-point numbers,  all values associated to the constant *name* in the header of the ephemeris file |eph|.    


.. include:: calceph.multiple_getconstantvd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getconstantss(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: str


.. include:: calceph.multiple_getconstantss.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvs|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getconstantvs(name) -> arrayvalue

    :param  str name: |arg_constant_name|
    :return: |arg_constant_arrayvalue|
    :rtype: list

This function returns, as strings of characters,  all values associated to the constant *name* in the header of the ephemeris file |eph|.    


.. include:: calceph.multiple_getconstantvs.rst


.. %------------------------------------------------

|menu_calceph_getconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getconstantcount ()

    :return:  |arg_constant_number|
    :rtype: int

.. include:: calceph.multiple_getconstantcount.rst

.. %------------------------------------------------

|menu_calceph_getconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getconstantindex (index) ->name, value

    :param  int index: |arg_constant_index|
    :return: |arg_constant_name|, |arg_constant_value|
    :rtype: str, float

.. include:: calceph.multiple_getconstantindex.rst

.. %------------------------------------------------

|menu_calceph_getfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getfileversion ()

    :return:  version of the ephemeris file
    :rtype: str

.. include:: calceph.multiple_getfileversion.rst

.. %------------------------------------------------

|menu_calceph_getidbyname|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getidbyname (name, unit) -> id

    :param  str name: |arg_name_getidbyname|
    :param  int unit: .. include:: arg_unit_idbyname.rst
    :return:  identification number of the body
    :rtype: int or None

This function returns the identification number of the body associated to the given name. If no such name exists in the mapping, then the value *None* is returned.

.. include:: calceph.multiple_getidbyname.rst

.. %------------------------------------------------

|menu_calceph_getnamebyidss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getnamebyidss (id, unit) -> name

    :param  int id: identification number of the body
    :param  int unit: .. include:: arg_unit_idbyname.rst
    :return: name of the body 
    :rtype: str or None

This function returns the first given name of the body associated to the identification number *id*. If no such name exists in the mapping, then the value *None* is returned.

.. include:: calceph.multiple_getnamebyidss.rst


.. %------------------------------------------------

|menu_calceph_gettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.gettimescale ()

    :return:  time scale of the ephemeris file
    :rtype: int

.. include:: calceph.multiple_gettimescale.rst

.. %------------------------------------------------

|menu_calceph_gettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.gettimespan () -> firsttime, lasttime, continuous

    :return:  first and last available time, availability of the quantities of the bodies over the time span
    :rtype: float, float, int



.. include:: calceph.multiple_gettimespan.rst


.. %------------------------------------------------

|menu_calceph_getpositionrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getpositionrecordcount ()

    :return:  |arg_positionrecordcount|
    :rtype: int


.. include:: calceph.multiple_getpositionrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getpositionrecordindex (index) -> target, center, firsttime, lasttime, frame

    :param  int index: |arg_positionrecord_index|
    :return:  .. include:: arg_positionrecordindex.rst
    :rtype: int, int, float, float, int



.. include:: calceph.multiple_getpositionrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getpositionrecordindex2 (index) -> target, center, firsttime, lasttime, frame, segid

    :param  int index: |arg_positionrecord_index|
    :return:  .. include:: arg_positionrecordindex2.rst
    :rtype: int, int, float, float, int, int

.. include:: calceph.multiple_getpositionrecordindex2.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getorientrecordcount ()

    :return:  |arg_orientrecordcount|
    :rtype: int

.. include:: calceph.multiple_getorientrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. py:function:: calcephpy.CalcephBin.getorientrecordindex (index) -> target, firsttime, lasttime, frame

    :param  int index: |arg_orientrecord_index|
    :return:  .. include:: arg_orientrecordindex.rst
    :rtype: int, float, float, int


.. include:: calceph.multiple_getorientrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getorientrecordindex2 (index) -> target, firsttime, lasttime, frame, segid

    :param  int index: |arg_orientrecord_index|
    :return:  .. include:: arg_orientrecordindex2.rst
    :rtype: int, float, float, int, int

.. include:: calceph.multiple_getorientrecordindex2.rst

.. %----------------------------------------------------------------------------

|menu_calceph_close|
~~~~~~~~~~~~~~~~~~~~


.. py:method:: calcephpy.CalcephBin.close ()


This function closes the access associated to |ephemerisdescriptoreph| and frees allocated memory for it.



