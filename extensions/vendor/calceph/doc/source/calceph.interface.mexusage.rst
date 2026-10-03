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

It is designed to work with Matlab or Octave software. 

With Octave, you should load this package :

.. code-block::  matlab
   
        pkg load calcephoct
        
If you want Octave to automatically load this package, simply add to the file *octaverc* the command **pkg load calcephoct** .


With Matlab, you should add the path to the Matlab files of the dynamic library |LIBRARYSHORTNAME| :

.. code-block::  matlab
    
        addpath('<prefix>/libexec/calceph/mex/')
        
By default, this prefix is */usr/local*, so you have to enter before using calceph library.        
        
.. code-block::  matlab
    
        addpath('/usr/local/libexec/calceph/mex/')


If you want Matlab to automatically add this path at startup, simply add to this path to the environment variable *MATLABPATH*.



.. If you receive a message message similar to ``error: ... : .../libexec/calceph/mex/interfacemex.mex: failed to load: libcalceph.so.1: cannot open shared object file: No such file or directory``, the path to the library is missing in the environment variable *LD_LIBRARY_PATH* or  *DYLD_LIBRARY_PATH* and you should add to it.

Relative to C or Fortran interface, the prefixes  *calceph_*, *CALCEPH_*, *NAIFID_* are deleted for the naming convention of the functions, constants and NAIF identification numbers.  


.. %----------------------------------------------------------------------------


.. _`NaifId`:

Types
=====

.. mat:class:: CalcephBin

This type contains all information to access a single ephemeris file or a set of files to be used together.

.. mat:class::  NaifId

This type contains the NAIF identification numbers.

.. mat:class::  Constants

This type contains all constants defined in the library, except the NAIF identification numbers.

.. %----------------------------------------------------------------------------

.. _`Constants`:

Constants
=========

The following constants are defined in the class **Constants**.


.. mat:attribute:: VERSION_MAJOR
    
This integer constant defines the major revision of this library. It can be used to distinguish different releases of this library.


.. mat:attribute:: VERSION_MINOR

This integer constant defines the minor revision of this library. It can be used to distinguish different releases of this library.


.. mat:attribute:: VERSION_PATCH
    
This integer constant defines the patch level revision of this library. It can be used to distinguish different releases of this library.

   

.. mat:attribute:: VERSION_STRING
    

This string is the version of the library, which can be compared to the result of calceph_getversion to check at run time if the header file and library used match:

Note: Obtaining different strings is not necessarily an error, as in general, a program compiled with some old CALCEPH version can be dynamically linked with a newer CALCEPH library version (if allowed by the operating system).


.. mat:attribute:: ASTEROID
    
This integer defines the offset value for the asteroids, starting at 2E6, that must be used as target or center for the computation functions, such as |calceph_compute|.

.. mat:attribute:: ASTEROID_8
    
This integer defines the offset value for the asteroids, starting at 2E8, that must be used as target or center for the computation functions, such as |calceph_compute|.


The following constants specify in which units are expressed the output of the computation functions, such as |calceph_compute_unit| :  

    
.. mat:attribute:: UNIT_AU
    
This integer defines that the unit of the positions and velocities is expressed in astronomical unit.


.. mat:attribute:: UNIT_KM
    
This integer defines that the unit of the positions and velocities is expressed in kilometer.


.. mat:attribute:: UNIT_DAY
    
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in day (one day=86400 seconds).


.. mat:attribute:: UNIT_SEC
    
This integer defines that the unit of the velocities or the quantity TT-TDB or TCG-TCB is expressed in second.

.. mat:attribute:: UNIT_RAD
    
This integer defines that the unit of the angles is expressed in radian.

    
.. mat:attribute:: OUTPUT_EULERANGLES
    
This integer defines that the output array contains the euler angles.

    
.. mat:attribute:: OUTPUT_NUTATIONANGLES

This integer defines that the output array contains the nutation angles.


.. mat:attribute:: USE_NAIFID
    
This integer defines that the NAIF identification numbers are used as target or center for the computation functions, such as |calceph_compute_unit|.

.. _`ConstantsSegType`:

The following constants specify the type of segments for the functions, such as |calceph_getmaxsupportedorder| :  


.. mat:attribute:: SEGTYPE_ORIG_0

This integer defines the type of segment for the original INPOP/JPL DE file format.


.. mat:attribute:: SEGTYPE_SPK_1
.. mat:attribute:: SEGTYPE_SPK_2
.. mat:attribute:: SEGTYPE_SPK_3
.. mat:attribute:: SEGTYPE_SPK_5
.. mat:attribute:: SEGTYPE_SPK_8
.. mat:attribute:: SEGTYPE_SPK_9
.. mat:attribute:: SEGTYPE_SPK_12
.. mat:attribute:: SEGTYPE_SPK_13
.. mat:attribute:: SEGTYPE_SPK_14
.. mat:attribute:: SEGTYPE_SPK_17
.. mat:attribute:: SEGTYPE_SPK_18
.. mat:attribute:: SEGTYPE_SPK_19
.. mat:attribute:: SEGTYPE_SPK_20
.. mat:attribute:: SEGTYPE_SPK_21
.. mat:attribute:: SEGTYPE_SPK_102
.. mat:attribute:: SEGTYPE_SPK_103
.. mat:attribute:: SEGTYPE_SPK_120

This integer defines the type of segments (|supportedspk|) for the SPICE Kernel files.


.. _`ConstantsTimescales`:

The following constants specify the time scale used for time conversion functions :

.. mat:attribute:: TDB
.. mat:attribute:: TCB
.. mat:attribute:: UTC
.. mat:attribute:: TAI
.. mat:attribute:: TT
.. mat:attribute:: TIMESCALE_FROM_STR

These integers define the timescales.


