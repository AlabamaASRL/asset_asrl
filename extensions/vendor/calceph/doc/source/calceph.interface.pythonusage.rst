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

It is designed to work with Python interpreters compliant with the Python 2.6 or later and Python 3.0 or later. 

All declarations needed to use |LIBRARYNAME| are collected in the module :file:`calcephpy`.
You should import this module:

    ::
    
        from calcephpy import *


If you receive the following message ``ImportError: No module named calcephpy`` and if the configuration option *enable-python-package-system* and *enable-python-package-user* was not set, the environment variable *PYTHONPATH* should be set to the right location of the |LIBRARYSHORTNAME| python package  ( e.g., PYTHONPATH=/usr/local/lib64/python3.4/site-packages/:$PYTHONPATH ) in your shell initialization file (e.g.,  ~/.bash_login or  ~/.profile), in order that the python interpreter finds the |LIBRARYSHORTNAME| python package. 

Relative to C or Fortran interface, the prefixes  *calceph_*, *CALCEPH_*, *NAIFID_* are deleted for the naming convention of the functions, constants and NAIF identification numbers.  


.. %----------------------------------------------------------------------------


.. _`NaifId`:

Types
=====


.. py:attribute:: calcephpy.CalcephBin

This type contains all information to access a single ephemeris file or a set of ephemeris files to be used together.

.. py:attribute::  calcephpy.NaifId

This type contains the NAIF identification numbers.

.. py:attribute::  calcephpy.Constants

This type contains all constants defined in the library, except the NAIF identification numbers.


.. %----------------------------------------------------------------------------

.. _`Constants`:

Constants
=========


The following constants are defined in the class **Constants** (or *calcephpy.Constants*).


.. py:data:: VERSION_MAJOR

    
This integer constant defines the major revision of this library. It can be used to distinguish different releases of this library.


.. py:data:: VERSION_MINOR
    

This integer constant defines the minor revision of this library. It can be used to distinguish different releases of this library.


.. py:data:: VERSION_PATCH
    
    
This integer constant defines the patch level revision of this library. It can be used to distinguish different releases of this library.



.. py:data:: VERSION_STRING
 
This string is the version of the library, which can be compared to the result of calceph_getversion to check at run time if the header file and library used match:

Note: Obtaining different strings is not necessarily an error, as in general, a program compiled with some old CALCEPH version can be dynamically linked with a newer CALCEPH library version (if allowed by the operating system).


.. py:data:: ASTEROID
 
This integer defines the offset value for the asteroids, starting at 2E6, that must be used as target or center for the computation functions, such as |calceph_compute|.

.. py:data:: ASTEROID_8
 
This integer defines the offset value for the asteroids, starting at 2E8, that must be used as target or center for the computation functions, such as |calceph_compute|.


The following constants specify in which units are expressed the output of the computation functions, such as |calceph_compute_unit| :  

    
.. py:data:: UNIT_AU
  
This integer defines that the unit of the positions and velocities is expressed in astronomical unit.



.. py:data:: UNIT_KM
  
    
This integer defines that the unit of the positions and velocities is expressed in kilometer.



.. py:data:: UNIT_DAY
    
    
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in day (one day=86400 seconds).


.. py:data:: UNIT_SEC
  
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in second.

.. py:data:: UNIT_RAD
    
    
This integer defines that the unit of the angles is expressed in radian.

   
.. py:data:: OUTPUT_EULERANGLES
  
This integer defines that the output array contains the euler angles.



.. py:data:: OUTPUT_NUTATIONANGLES


This integer defines that the output array contains the nutation angles.

.. py:data:: USE_NAIFID
 
    
This integer defines that the NAIF identification numbers are used as target or center for the computation functions, such as |calceph_compute_unit|.

.. _`ConstantsSegType`:

The following constants specify the type of segments for the functions, such as |calceph_getmaxsupportedorder| :  

.. py:data:: SEGTYPE_ORIG_0
    
This integer defines the type of segment for the original INPOP/JPL DE file format.


.. py:data:: SEGTYPE_SPK_1
.. py:data:: SEGTYPE_SPK_2
.. py:data:: SEGTYPE_SPK_3
.. py:data:: SEGTYPE_SPK_5
.. py:data:: SEGTYPE_SPK_8
.. py:data:: SEGTYPE_SPK_9
.. py:data:: SEGTYPE_SPK_12
.. py:data:: SEGTYPE_SPK_13
.. py:data:: SEGTYPE_SPK_14
.. py:data:: SEGTYPE_SPK_17
.. py:data:: SEGTYPE_SPK_18
.. py:data:: SEGTYPE_SPK_19
.. py:data:: SEGTYPE_SPK_20
.. py:data:: SEGTYPE_SPK_21
.. py:data:: SEGTYPE_SPK_102
.. py:data:: SEGTYPE_SPK_103
.. py:data:: SEGTYPE_SPK_120

This integer defines the type of segments (|supportedspk|) for the SPICE Kernel files.


.. _`ConstantsTimescales`:

The following constants specify the time scale used for time conversion functions :

.. py:data:: TDB
.. py:data:: TCB
.. py:data:: UTC
.. py:data:: TAI
.. py:data:: TT
.. py:data:: TIMESCALE_FROM_STR

These integers define the timescales.

