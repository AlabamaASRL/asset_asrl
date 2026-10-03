
.. include:: replace.rst

Miscellaneous functions
=======================

|menu_calceph_getmaxsupportedorder|
-----------------------------------

.. py:method:: calcephpy.getmaxsupportedorder (segid)  

    :param  int segid: |arg_segid|
    :return: |arg_version|
    :rtype: str 


This function returns the maximal order of the derivatives computed by the functions |calceph_compute_order|, |calceph_orient_order|, ....  for the segment type *segid*.
If the segment type is unknown by the library, the function returns -1.


The accepted values of *segid** are the predefined constants *Constants.SEGTYPE_...* (:ref:`Constants`).


::

    from calcephpy import *
    maxorder = getmaxsupportedorder(Constants.SEGTYPE_SPK_2)
    print('maximal order is ', maxorder)



|menu_calceph_getversion_str|
-----------------------------

.. py:method:: calcephpy.getversion_str () 

    :return: |arg_version|
    :rtype: str 



This function returns the version of the |LIBRARYNAME|, as a string.

::

    from calcephpy import *
    print('version=', getversion_str())


