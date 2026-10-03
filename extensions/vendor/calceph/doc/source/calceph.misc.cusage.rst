
.. include:: replace.rst

Miscellaneous functions
=======================


|menu_calceph_getmaxsupportedorder|
-----------------------------------


.. c:function:: int calceph_getmaxsupportedorder (int segid)

    :param  segid: |arg_segid|
    :return: maximal order of the computable derivatives for this type of segment.



This function returns the maximal order of the derivatives computed by the functions |calceph_compute_order|, |calceph_orient_order|, ....  for the segment type *segid*.
If the segment type is unknown by the library, the function returns -1.


The accepted values of *segid* are the predefined constants *CALCEPH_SEGTYPE_...* (:ref:`Constants`).

::
            
    int maxorder = calceph_getmaxsupportedorder(CALCEPH_SEGTYPE_SPK_2);
    printf ("maximal order is %d \n", maxorder);



|menu_calceph_getversion_str|
-----------------------------


.. c:function:: void calceph_getversion_str ( char version[CALCEPH_MAX_CONSTANTNAME])

    :param  version: |arg_version|



This function returns the version of the |LIBRARYNAME|, as a null-terminated string.

::
            
    char cversion[CALCEPH_MAX_CONSTANTNAME];
    calceph_getversion_str(cversion);
    printf ("library version is '%s'\n", cversion);


