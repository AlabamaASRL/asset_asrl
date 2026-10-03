.. include:: replace.rst

Error functions
===============

The following group of functions defines the behavior of the library when errors occur during the execution. 


.. %----------------------------------------------------------------------------

Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 


The example in Python language is :file:`pyerror.py`.


.. include:: examples/error_usage.rst


.. %----------------------------------------------------------------------------

|menu_calceph_seterrorhandler|
------------------------------

.. py:method:: calcephpy.seterrorhandler (typehandler, userfunc) 

    :param  int typehandler: |arg_typehandler|
    :param  function userfunc: |arg_userfunc|

.. include:: calceph.error.rst

If the function is called with 1 or 2 for *typehandler*, the parameter *userfunc* must be set to *0*.


The function *userfunc* must be defined as 

::

    def userfunc (msg)
    # parameter msg is of type str
