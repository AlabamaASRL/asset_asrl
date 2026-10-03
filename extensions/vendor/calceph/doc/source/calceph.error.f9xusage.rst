.. include:: replace.rst

Error functions
===============

The following group of functions defines the behavior of the library when errors occur during the execution. 


.. %----------------------------------------------------------------------------

Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 


The example in Fortran 77/90/95 language is :file:`f77error.f`.


.. include:: examples/error_usage.rst


.. %----------------------------------------------------------------------------

|menu_calceph_seterrorhandler|
------------------------------

.. f:subroutine:: f90calceph_seterrorhandler (typehandler, userfunc)

    :param  typehandler [INTEGER, intent(in)]: |arg_typehandler|
    :param  userfunc [EXTERNAL, intent(in)]: |arg_userfunc|


.. include:: calceph.error.rst


If the function is called with 1 or 2 for *typehandler*, the parameter *userfunc* is ignored and it may be an empty function.

The function *userfunc* must be defined as 

::

    subroutine userfunc (msg) 
    implicit none
    CHARACTER(len=*), intent(in) :: msg

This function must be declared as **EXTERNAL**
