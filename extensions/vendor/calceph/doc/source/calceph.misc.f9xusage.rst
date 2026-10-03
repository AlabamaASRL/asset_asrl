
.. include:: replace.rst

Miscellaneous functions
=======================


|menu_calceph_getmaxsupportedorder|
-----------------------------------

.. f:subroutine:: f90calceph_getmaxsupportedorder (segid) 

    :param  version [INTEGER, intent(in)]: |arg_segid|
    :r f90calceph_getmaxsupportedorder: maximal order of the computable derivatives for this type of segment.
    :rtype f90calceph_getmaxsupportedorder: INTEGER


This function returns the maximal order of the derivatives computed by the functions |calceph_compute_order|, |calceph_orient_order|, ....  for the segment type *segid*.
If the segment type is unknown by the library, the function returns -1.


The accepted values of *segid* are the predefined constants *CALCEPH_SEGTYPE_...* (:ref:`Constants`).


::

    integer maxorder
    
    maxorder = calceph_getmaxsupportedorder(CALCEPH_SEGTYPE_SPK_2)
    write(*,*) 'maximal order is ', maxorder



|menu_calceph_getversion_str|
-----------------------------

.. f:subroutine:: f90calceph_getversion_str (version) 

    :param  version [CHARACTER(len=CALCEPH_MAX_CONSTANTNAME), intent(out)]: |arg_version|


This function returns the version of the |LIBRARYNAME|, as a string.


Trailing blanks are added to the name version.




::

    character(len=CALCEPH_MAX_CONSTANTNAME) version
    
    call f90calceph_getversion_str(version)
    write(*,*) 'library version is ', version
