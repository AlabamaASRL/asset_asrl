.. _`Reading/Evaluation functions`:

.. include:: calceph.multiple_intro.rst


.. @c %----------------------------------------------------------------------------

Functions
---------

.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

|menu_calceph_open|
~~~~~~~~~~~~~~~~~~~


.. mat:staticmethod:: CalcephBin.open (filename) -> eph

    :param  str filename: |arg_filename|
    :return: |arg_eph|
    :rtype: CalcephBin 

.. include:: calceph.multiple_open.rst

.. %----------------------------------------------------------------------------

|menu_calceph_open_array|
~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:staticmethod:: CalcephBin.open ( array_filename ) -> eph

    :param  list array_filename: |arg_array_filename|
    :return: |arg_eph|
    :rtype: CalcephBin 

.. include:: calceph.multiple_open_array.rst

.. %----------------------------------------------------------------------------

|menu_calceph_prefetch|
~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.prefetch()


.. include:: calceph.multiple_prefetch.rst

.. %----------------------------------------------------------------------------

|menu_calceph_isthreadsafe|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.isthreadsafe()


.. include:: calceph.multiple_isthreadsafe.rst


.. %----------------------------------------------------------------------------

|menu_calceph_compute|
~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.compute (JD0, time, target, center) -> PV

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target|
    :param  int center: |arg_center|
    :return:  .. include:: arg_PV.rst
    :rtype: vector


.. include:: calceph.multiple_compute.rst

   
.. %----------------------------------------------------------------------------

|menu_calceph_compute_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.compute_unit (JD0, time, target, center, unit) -> PV

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_unit|
    :param  int center: |arg_center_unit|
    :param  int unit: .. include:: arg_unit.rst
    :return:  .. include:: arg_PV_unit.rst
    :rtype: vector

.. include:: calceph.multiple_compute_unit.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.orient_unit (JD0, time, target,  unit) -> PV

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :return:  .. include:: arg_PV_orient_unit.rst
    :rtype: vector


.. include:: calceph.multiple_orient_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_unit|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: calcephpy.CalcephBin.rotangmom_unit (JD0, time, target, unit) -> PV

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :return:  .. include:: arg_PV_rotangmom_unit.rst
    :rtype: vector

.. include:: calceph.multiple_rotangmom_unit.rst

.. %----------------------------------------------------------------------------

|menu_calceph_compute_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.compute_order(JD0, time, target, center, unit, order) -> PVAJ

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_unit|
    :param  int center: |arg_center_unit|
    :param  int unit: .. include:: arg_unit_order.rst
    :param  int order: .. include:: arg_order.rst
    :return:  .. include:: arg_PVAJ_order.rst
    :rtype: vector

.. include:: calceph.multiple_compute_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_orient_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.orient_order(JD0, time, target, unit, order) -> PVAJ

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :param  int order: .. include:: arg_order_orient.rst
    :return: .. include:: arg_PVAJ_orient_order.rst
    :rtype: vector


.. include:: calceph.multiple_orient_order.rst

.. %----------------------------------------------------------------------------

|menu_calceph_rotangmom_order|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.rotangmom_order(JD0, time, target, unit, order) -> PVAJ

    :param  double JD0: |arg_JD0|
    :param  double time: |arg_time|
    :param  int target: |arg_target_orient_unit|
    :param  int unit: .. include:: arg_unit_orient_unit.rst
    :param  int order: .. include:: arg_order_rotangmom.rst
    :return: .. include:: arg_PVAJ_rotangmom_order.rst
    :rtype: vector


.. include:: calceph.multiple_rotangmom_order.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getconstant(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: double

.. include:: calceph.multiple_getconstant.rst


.. %----------------------------------------------------------------------------

|menu_calceph_getconstantsd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getconstantsd(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: double


.. include:: calceph.multiple_getconstantsd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvd|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getconstantvd(name) -> arrayvalue

    :param  str name: |arg_constant_name|
    :return: |arg_constant_arrayvalue|
    :rtype: vector

This function returns, as floating-point numbers, all values associated to the constant *name* in the header of the ephemeris file |eph|. 


.. include:: calceph.multiple_getconstantvd.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getconstantss(name) -> value

    :param  str name: |arg_constant_name|
    :return: |arg_constant_value|
    :rtype: string


.. include:: calceph.multiple_getconstantss.rst

.. %----------------------------------------------------------------------------

|menu_calceph_getconstantvs|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getconstantvs(name) -> arrayvalue

    :param  str name: |arg_constant_name|
    :return: |arg_constant_arrayvalue|
    :rtype: cell array of character vectors

This function returns, as strings of characters, all values associated to the constant *name* in the header of the ephemeris file |eph|.


.. include:: calceph.multiple_getconstantvs.rst


.. %------------------------------------------------

|menu_calceph_getconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getconstantcount ()

    :return:  |arg_constant_number|
    :rtype: int

.. include:: calceph.multiple_getconstantcount.rst

.. %------------------------------------------------

|menu_calceph_getconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getconstantindex (index) ->name, value

    :param  int index: |arg_constant_index|
    :return: |arg_constant_name|, |arg_constant_value|
    :rtype: str, double


.. include:: calceph.multiple_getconstantindex.rst

.. %------------------------------------------------

|menu_calceph_getfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getfileversion ()

    :return:  version of the ephemeris file
    :rtype: str

.. include:: calceph.multiple_getfileversion.rst

.. %------------------------------------------------

|menu_calceph_getidbyname|
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getidbyname (name, unit) -> found, id

    :param  str name: |arg_name_getidbyname|
    :param  int unit: .. include:: arg_unit_idbyname.rst
    :return:  avaibility of the id, identification number of the body
    :rtype: int, int

This function returns the identification number of the body associated to the given name. 

It returns the following value in the parameter *found* :

* 0 if no such name exists in the mapping. The value of *id* is undefined. 
* 1 if the name exists in the mapping. The value of *id* is valid. 


.. include:: calceph.multiple_getidbyname.rst


.. %------------------------------------------------

|menu_calceph_getnamebyidss|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getnamebyidss (id, unit) -> found, name

    :param  int id: identification number of the body
    :param  int unit: .. include:: arg_unit_idbyname.rst
    :return:  avaibility of the name, name of the body
    :rtype: int, str

This function returns the first given name of the body associated to the identification number *id*. 

It returns the following value in the parameter *found* :

* 0 if no such identification number exists in the mapping. The value of *name* is undefined. 
* 1 if the identification number *id* exists in the mapping. The value of *name* is valid.


.. include:: calceph.multiple_getnamebyidss.rst


.. %------------------------------------------------

|menu_calceph_gettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.gettimescale ()

    :return:  time scale of the ephemeris file
    :rtype: int

.. include:: calceph.multiple_gettimescale.rst

.. %------------------------------------------------

|menu_calceph_gettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.gettimespan () -> firsttime, lasttime, continuous

    :return:  first and last available time, availability of the quantities of the bodies over the time span
    :rtype: double, double, int


.. include:: calceph.multiple_gettimespan.rst


.. %------------------------------------------------

|menu_calceph_getpositionrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: calcephpy.CalcephBin.getpositionrecordcount ()

    :return:  |arg_positionrecordcount|
    :rtype: int

.. include:: calceph.multiple_getpositionrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getpositionrecordindex (index) -> target, center, firsttime, lasttime, frame

    :param  int index: |arg_positionrecord_index|
    :return:  .. include:: arg_positionrecordindex.rst
    :rtype: int, int, double, double, int


.. include:: calceph.multiple_getpositionrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getpositionrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getpositionrecordindex2 (index) -> target, center, firsttime, lasttime, frame,segid

    :param  int index: |arg_positionrecord_index|
    :return:  .. include:: arg_positionrecordindex2.rst
    :rtype: int, int, double, double, int, int


.. include:: calceph.multiple_getpositionrecordindex2.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getorientrecordcount ()

    :return:  |arg_orientrecordcount|
    :rtype: int

.. include:: calceph.multiple_getorientrecordcount.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. mat:method:: CalcephBin.getorientrecordindex (index) -> target, firsttime, lasttime, frame

    :param  int index: |arg_orientrecord_index|
    :return:  .. include:: arg_orientrecordindex.rst
    :rtype: int, double, double, int


.. include:: calceph.multiple_getorientrecordindex.rst

.. %------------------------------------------------

|menu_calceph_getorientrecordindex2|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getorientrecordindex2 (index) -> target, firsttime, lasttime, frame, segid

    :param  int index: |arg_orientrecord_index|
    :return:  .. include:: arg_orientrecordindex2.rst
    :rtype: int, double, double, int, int

.. include:: calceph.multiple_getorientrecordindex2.rst

.. %----------------------------------------------------------------------------

|menu_calceph_close|
~~~~~~~~~~~~~~~~~~~~

.. mat:method:: calcephpy.CalcephBin.close ()

This function closes the access associated to |ephemerisdescriptoreph| and frees allocated memory for it.



