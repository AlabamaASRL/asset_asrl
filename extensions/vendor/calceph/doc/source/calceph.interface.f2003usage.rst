Library interface
*****************

A simple example program
========================
    
The following example program shows the typical usage of the |API| interface.

Other examples using the |API| interface can be found in the directory *examples* of the library sources. 

.. include:: examples/simple_program.rst

.. _calcephpy:


|menu_Headers_and_Libraries|
============================

All declarations needed to use |LIBRARYNAME| are collected in the module files :file:`calceph.mod`.  The library is designed to work with Fortran compilers compliant with the Fortran 2003 standard. All declarations use the standard **ISO_C_BINDING** module.

You should include that module in any program using the |LIBRARYSHORTNAME| library:

::

    use calceph


When a fortran string is given as a parameter to a function of this library, you should append this string with **//C_NULL_CHAR** because the C library works only with C string.
   
.. highlight::  bash

Compilation on a Unix-like system
---------------------------------
 
 
 All programs using  |LIBRARYSHORTNAME| must link against the :file:`libcalceph` library.  On Unix-like system this can be done with *-lcalceph*, for example

    ::
  
        gfortran -I/usr/local/include myprogram.f -o myprogram -lcalceph 


 If  |LIBRARYNAME| has been installed to a non-standard location then it may be necessary to use *-I* and *-L* compiler options to point to the right directories, and some sort of run-time path for a shared library.

Compilation on a Windows system
-------------------------------

 All programs using |LIBRARYSHORTNAME| must link against the :file:`libcalceph.lib`.  On Windows system this can be done with *libcalceph.lib*, for example

    ::

        gfortran.exe /out:myprogram.exe myprogram.f libcalceph.lib 

 If  |LIBRARYNAME| has been installed to a non-standard location then it may be necessary to use */I* and */LIBPATH:* compiler options to point to the right directories.

.. highlight::  none


.. %----------------------------------------------------------------------------

.. _`Constants`:

Constants
=========


The following constants are defined in the module :file:`calceph.mod`.
 

.. f:variable:: CALCEPH_MAX_CONSTANTNAME
    :type: integer

This integer defines the maximum number of characters, including the trailing '\\0',  that the name of a constant, available from the ephemeris file, could contain.



.. f:variable:: CALCEPH_MAX_CONSTANTVALUE
    :type: integer

This integer defines the maximum number of characters, including the trailing '\\0',  that the value of a constant, available from the ephemeris file, could contain if the value is stored as a string of characters.

.. f:variable:: CALCEPH_VERSION_MAJOR
    :type: integer

    
This integer constant defines the major revision of this library. It can be used to distinguish different releases of this library.

.. f:variable:: CALCEPH_VERSION_MINOR
    :type: integer


This integer constant defines the minor revision of this library. It can be used to distinguish different releases of this library.

.. f:variable:: CALCEPH_VERSION_PATCH
    :type: integer

    
This integer constant defines the patch level revision of this library. It can be used to distinguish different releases of this library.


.. f:variable:: CALCEPH_VERSION_STRING
    :type: character(len=*)



This string is the version of the library, which can be compared to the result of calceph_getversion to check at run time if the header file and library used match:

Note: Obtaining different strings is not necessarily an error, as in general, a program compiled with some old CALCEPH version can be dynamically linked with a newer CALCEPH library version (if allowed by the operating system).



.. f:variable:: CALCEPH_ASTEROID
    :type: integer


This integer defines the offset value for the asteroids, starting at 2E6, that must be used as target or center for the computation functions, such as |calceph_compute|.


.. f:variable:: CALCEPH_ASTEROID_8
    :type: integer


This integer defines the offset value for the asteroids, starting at 2E8, that must be used as target or center for the computation functions, such as |calceph_compute|.


The following constants specify in which units are expressed the output of the computation functions, such as |calceph_compute_unit| :  


.. f:variable:: CALCEPH_UNIT_AU
    :type: integer

    
This integer defines that the unit of the positions and velocities is expressed in astronomical unit.

.. f:variable:: CALCEPH_UNIT_KM
    :type: integer

This integer defines that the unit of the positions and velocities is expressed in kilometer.


.. f:variable:: CALCEPH_UNIT_DAY
    :type: integer

This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in day (one day=86400 seconds).


.. f:variable:: CALCEPH_UNIT_SEC
    :type: integer

This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in second.


.. f:variable:: CALCEPH_UNIT_RAD
    :type: integer


This integer defines that the unit of the angles is expressed in radian.


.. f:variable:: CALCEPH_OUTPUT_EULERANGLES
    :type: integer

    
This integer defines that the output array contains the euler angles.


.. f:variable:: CALCEPH_OUTPUT_NUTATIONANGLES
    :type: integer


This integer defines that the output array contains the nutation angles.


.. f:variable:: CALCEPH_USE_NAIFID
    :type: integer

    
This integer defines that the NAIF identification numbers are used as target or center for the computation functions, such as |calceph_compute_unit|.

.. _`ConstantsSegType`:

The following constants specify the type of segments for the functions, such as |calceph_getmaxsupportedorder| :  


.. f:variable:: CALCEPH_SEGTYPE_ORIG_0
    :type: integer


This integer defines the type of segment for the original INPOP/JPL DE file format.

.. f:variable:: CALCEPH_SEGTYPE_SPK_1
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_2
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_3
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_5
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_8
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_9
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_12
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_13
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_14
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_17
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_18
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_19
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_20
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_21
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_102
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_103
    :type: integer
.. f:variable:: CALCEPH_SEGTYPE_SPK_120
    :type: integer


This integer defines the type of segments (|supportedspk|) for the SPICE Kernel files.

.. _`ConstantsTimescales`:


The following constants specify the time scale used for time conversion functions:

.. f:variable:: CALCEPH_TDB
    :type: integer
.. f:variable::  CALCEPH_TCB
    :type: integer
.. f:variable:: CALCEPH_UTC
    :type: integer
.. f:variable:: CALCEPH_TAI
    :type: integer
.. f:variable:: CALCEPH_TT
    :type: integer
.. f:variable:: CALCEPH_TIMESCALE_FROM_STR
    :type: integer
