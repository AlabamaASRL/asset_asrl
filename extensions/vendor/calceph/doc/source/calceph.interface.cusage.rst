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

All declarations needed to use |LIBRARYNAME| are collected in the include file
:file:`calceph.h`.  It is designed to work with both C and C++ compilers.

You should include that file in any program using the |LIBRARYSHORTNAME| library:

::

    #include <calceph.h>


.. highlight::  bash

Compilation on a Unix-like system
---------------------------------
 
All programs using  |LIBRARYSHORTNAME| must link against the :file:`libcalceph` library.  On Unix-like system this can be done with *-lcalceph*, for example

    
    ::
   
        gcc myprogram.c -o myprogram -lcalceph 
     

If  |LIBRARYNAME| has been installed to a non-standard location then it may be necessary to use *-I* and *-L* compiler options to point to the right directories, and some sort of run-time path for a shared library.

Compilation on a Windows system
-------------------------------
 
* Using the Windows SDK

 All programs using |LIBRARYSHORTNAME| must link against the :file:`libcalceph.lib`.  On Windows system this can be done with *libcalceph.lib*, for example

    ::
    
        cl.exe /out:myprogram myprogram.c libcalceph.lib 


 If  |LIBRARYNAME| has been installed to a non-standard location then it may be necessary to use */I* and */LIBPATH:* compiler options to point to the right directories.

* Using the MinGW
 
 All programs using  |LIBRARYSHORTNAME| must link against the :file:`libcalceph` library.  On the MinGW system, this can be done with *-lcalceph*, for example

    ::
    
        gcc.exe myprogram.c -o myprogram -lcalceph 

 If  |LIBRARYNAME| has been installed to a non-standard location then it may be necessary to use *-I* and *-L* compiler options to point to the right directories, and some sort of run-time path for a shared library.

.. highlight::  none


.. %----------------------------------------------------------------------------


.. _`NaifId`:

Types
=====


.. c:type:: t_calcephbin

This type contains all information to access a single ephemeris file or a set of files to be used together.

.. c:type:: t_calceph_charvalue

This type is a array of characters to store the value of the constants as a string.

.. c:type:: t_writephbin

This type contains all information to write to an ephemeris file.

.. %----------------------------------------------------------------------------

.. _`Constants`:

Constants
=========


.. c:macro:: CALCEPH_MAX_CONSTANTNAME

This integer defines the maximum number of characters, including the trailing '\\0',  that the name of a constant, available from the ephemeris file, could contain.

.. c:macro:: CALCEPH_MAX_CONSTANTVALUE

This integer defines the maximum number of characters, including the trailing '\\0',  that the value of a constant, available from the ephemeris file, could contain if the value is stored as a string of characters.

.. c:macro:: CALCEPH_VERSION_MAJOR
    
This integer constant defines the major revision of this library. It can be used to distinguish different releases of this library.

.. c:macro:: CALCEPH_VERSION_MINOR

This integer constant defines the minor revision of this library. It can be used to distinguish different releases of this library.

.. c:macro:: CALCEPH_VERSION_PATCH

This integer constant defines the patch level revision of this library. It can be used to distinguish different releases of this library.


::

    #if   (CALCEPH_VERSION_MAJOR>=2) 
        ||  (CALCEPH_VERSION_MAJOR>=3 && CALCEPH_VERSION_MINOR>=2)
    ...
    #endif


.. c:macro:: CALCEPH_VERSION_STRING

This C null-terminated string constant is the version of the library, which can be compared to the result of calceph_getversion to check at run time if the header file and library used match:

::

    char version[CALCEPH_MAX_CONSTANTNAME];
    calceph_getversion_str(version);
    if (strcmp (version, CALCEPH_VERSION_STRING)!=0)
    fprintf (stderr, "Warning: header and library do not match\n");


Note: Obtaining different strings is not necessarily an error, as in general, a program compiled with some old CALCEPH version can be dynamically linked with a newer CALCEPH library version (if allowed by the operating system).


.. c:macro:: CALCEPH_ASTEROID

    
This integer defines the offset value for the asteroids, starting at 2E6, that must be used as target or center for the computation functions, such as |calceph_compute|.


.. c:macro:: CALCEPH_ASTEROID_8

    
This integer defines the offset value for the asteroids, starting at 2E8, that must be used as target or center for the computation functions, such as |calceph_compute|.



The following constants specify in which units are expressed the output of the computation functions, such as |calceph_compute_unit| :  


.. c:macro:: CALCEPH_UNIT_AU

    
This integer defines that the unit of the positions and velocities is expressed in astronomical unit.


.. c:macro:: CALCEPH_UNIT_KM
    
This integer defines that the unit of the positions and velocities is expressed in kilometer.

.. c:macro:: CALCEPH_UNIT_DAY

    
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in day (one day=86400 seconds).


.. c:macro:: CALCEPH_UNIT_SEC

    
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in second.


.. c:macro:: CALCEPH_UNIT_RAD

    
This integer defines that the unit of the angles is expressed in radian.


.. c:macro:: CALCEPH_OUTPUT_EULERANGLES

    
This integer defines that the output array contains the euler angles.


.. c:macro:: CALCEPH_OUTPUT_NUTATIONANGLES


This integer defines that the output array contains the nutation angles.


.. c:macro:: CALCEPH_USE_NAIFID

    
This integer defines that the NAIF identification numbers are used as target or center for the computation functions, such as |calceph_compute_unit|.

.. _`ConstantsSegType`:

The following constants specify the type of segments for the functions, such as |calceph_getmaxsupportedorder| :  


.. c:macro:: CALCEPH_SEGTYPE_ORIG_0


This integer defines the type of segment for the original INPOP/JPL DE file format.


.. c:macro:: CALCEPH_SEGTYPE_SPK_1
.. c:macro:: CALCEPH_SEGTYPE_SPK_2
.. c:macro:: CALCEPH_SEGTYPE_SPK_3
.. c:macro:: CALCEPH_SEGTYPE_SPK_5
.. c:macro:: CALCEPH_SEGTYPE_SPK_8
.. c:macro:: CALCEPH_SEGTYPE_SPK_9
.. c:macro:: CALCEPH_SEGTYPE_SPK_12
.. c:macro:: CALCEPH_SEGTYPE_SPK_13
.. c:macro:: CALCEPH_SEGTYPE_SPK_14
.. c:macro:: CALCEPH_SEGTYPE_SPK_17
.. c:macro:: CALCEPH_SEGTYPE_SPK_18
.. c:macro:: CALCEPH_SEGTYPE_SPK_19
.. c:macro:: CALCEPH_SEGTYPE_SPK_20
.. c:macro:: CALCEPH_SEGTYPE_SPK_21
.. c:macro:: CALCEPH_SEGTYPE_SPK_102
.. c:macro:: CALCEPH_SEGTYPE_SPK_103
.. c:macro:: CALCEPH_SEGTYPE_SPK_120


This integer defines the type of segments (|supportedspk|) for the SPICE Kernel files.

.. _`ConstantsTimescales`:

The following constants specify the  time scale used for time conversion functions:

.. c:macro:: CALCEPH_TDB
.. c:macro:: CALCEPH_TCB
.. c:macro:: CALCEPH_UTC
.. c:macro:: CALCEPH_TAI
.. c:macro:: CALCEPH_TT
.. c:macro:: CALCEPH_TIMESCALE_FROM_STR
