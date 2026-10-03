.. include:: replace.rst

Error functions
===============

The following group of functions defines the behavior of the library when errors occur during the execution. 


.. %----------------------------------------------------------------------------

Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 

The example in C language is :file:`cerror.c`. 

.. include:: examples/error_usage.rst


.. %----------------------------------------------------------------------------

|menu_calceph_seterrorhandler|
------------------------------


.. c:function:: void calceph_seterrorhandler (int typehandler, void (*userfunc)(const char*) )

    :param  typehandler: |arg_typehandler|
    :param  userfunc: |arg_userfunc|


.. include:: calceph.error.rst


If the function is called with 1 or 2 for *typehandler*, the parameter *userfunc* must be set to *NULL*.
