.. include:: replace.rst

Error functions
===============

The following group of functions defines the behavior of the library when errors occur during the execution. 


.. %----------------------------------------------------------------------------

Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 

The example in Octave/Matlab language is :file:`mexerror.m`.

.. include:: examples/error_usage.rst


.. %----------------------------------------------------------------------------

|menu_calceph_seterrorhandler|
------------------------------

.. mat:staticmethod:: calceph_seterrorhandler (typehandler, userfunc) 

    :param  int typehandler: |arg_typehandler|
    :param  function userfunc: |arg_userfunc| string

.. include:: calceph.error.rst



If the function is called with 1 or 2 for *typehandler*, the parameter *userfunc* must be set to an empty string ''.


The function *userfunc* must be defined as 

::

    function userfunc (msg)
    % parameter msg is of type string
    end 
