.. include:: replace.rst

Error functions
===============

The following group of functions defines the behavior of the library when errors occur during the execution. 


.. %----------------------------------------------------------------------------

Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 

The example in Fortran 2003 language is :file:`f2003error.f`.
    

.. include:: examples/error_usage.rst


.. %----------------------------------------------------------------------------

|menu_calceph_seterrorhandler|
------------------------------


.. f:subroutine:: calceph_seterrorhandler (typehandler, userfunc ) BIND(C)

    :param  typehandler [TYPE(C_INT), VALUE, intent(in)]: |arg_typehandler|
    :param  userfunc [TYPE(C_FUNPTR), VALUE, intent(in)]: |arg_userfunc|

.. include:: calceph.error.rst



If the function is called with 1 or 2 for *typehandler*, the parameter *userfunc* must be set to *C_NULL_FUNPTR*.

The function *userfunc* must be defined as 

::

    subroutine userfunc (msg, msglen)  BIND(C)
    USE, INTRINSIC :: ISO_C_BINDING
    implicit none
    CHARACTER(kind=C_CHAR), dimension(msglen), intent(in) :: msg
    INTEGER(C_INT), VALUE, intent(in) :: msglen

This function must have an explicit interface. 
