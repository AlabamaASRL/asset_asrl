
.. include:: replace.rst

Miscellaneous functions
=======================

|menu_calceph_getmaxsupportedorder|
-----------------------------------


.. mat:function:: calceph_getmaxsupportedorder (segid) 

    :param  int arg_segid: |arg_segid|
    :return: maximal order of the computable derivatives for this type of segment.
    :rtype: int


This function returns the maximal order of the derivatives computed by the functions |calceph_compute_order|, |calceph_orient_order|, ....  for the segment type *segid*.
If the segment type is unknown by the library, the function returns -1.


The accepted values of *segid** are the predefined constants *Constants.SEGTYPE_...* (:ref:`Constants`).



::

    maxorder = calceph_getmaxsupportedorder(Constants.SEGTYPE_SPK_2)


|menu_calceph_getversion_str|
-----------------------------


.. mat:function:: calceph_getversion_str () 

    :returns: |arg_version|
    :rtype: str 



This function returns the version of the |LIBRARYNAME|, as a string.




::

    version = calceph_getversion_str()

