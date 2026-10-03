
Reading/Evaluation functions
============================

The following group of functions should be the preferred method to access to the library. They allow to access to multiple ephemeris files at the same time, even  by multiple threads.  

When an error occurs, these functions execute error handlers according to the behavior defined by the function |calceph_seterrorhandler|. 

.. %----------------------------------------------------------------------------

Time notes
----------

The functions |calceph_compute|, |calceph_compute_unit|,  |calceph_compute_order|, |calceph_orient_unit|, ...  only accept a date expressed in the same time scale as the ephemeris files, which can be retrieved using the function |calceph_gettimescale|. Ephemeris files are generally expressed using the time scale TDB.
If a date, expressed in the TT (Terrestrial Time) time scale, is supplied to them, these functions will return an erroneous position of the order of several tens of meters for the planets.
If a date, expressed in the Coordinated Universal Time (UTC), is supplied to them, these functions  will return a very large erroneous position over several thousand kilometers for the planets.


Thread notes
------------

If the standard I/O functions such as **fread** are not reentrant
then the |LIBRARYSHORTNAME| I/O functions using them will not be reentrant either.
     
.. ifconfig:: calcephapi in ('C')

    It's safe for two threads to call the functions with the same object of type :c:type:`t_calcephbin` if and only if the function |calceph_isthreadsafe| returns a non-zero value. A previous call to the function |calceph_prefetch| is required for the function |calceph_isthreadsafe| to return a non-zero value.
    
    It's safe for two threads to access simultaneously to the same ephemeris file with two different objects of type :c:type:`t_calcephbin`. In this case, each thread must open the same file. 

.. ifconfig:: calcephapi in ('F2003', 'F90')

    It's safe for two threads to call the functions with the same handle of ephemeris object if and only if the function |calceph_isthreadsafe| returns a non-zero value.  A previous call to the function |calceph_prefetch| is required for the function |calceph_isthreadsafe| to return a non-zero value.
    
    It's safe for two threads to access simultaneously to the same ephemeris file with two different objects. In this case, each thread must open the same file. 


.. ifconfig:: calcephapi in ('Python')

    It's not safe for two threads to call the functions with the same object of type :py:class:`CalcephBin` if and only if the function |calceph_isthreadsafe| returns a non-zero value.  A previous call to the function |calceph_prefetch| is required for the function |calceph_isthreadsafe| to return a non-zero value.
    
    It's safe for two threads to access simultaneously to the same ephemeris file with two different objects of type :py:class:`CalcephBin`. In this case, each thread must open the same file. 

.. ifconfig:: calcephapi in ('Mex')

    It's not safe for two threads to call the functions with the same object of type :mat:class:`CalcephBin` if and only if the function |calceph_isthreadsafe| returns a non-zero value.  A previous call to the function |calceph_prefetch| is required for the function |calceph_isthreadsafe| to return a non-zero value.
    
    It's safe for two threads to access simultaneously to the same ephemeris file with two different objects of type :mat:class:`CalcephBin`. In this case, each thread must open the same file. 


Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions.

.. ifconfig:: calcephapi in ('C')

    The example in C language is :file:`cmultiple.c`. 
    
.. ifconfig:: calcephapi in ('F2003')

    The example in Fortran 2003 language is :file:`f2003multiple.f`.
    
.. ifconfig:: calcephapi in ('F90')

    The example in Fortran 77/90/95 language is :file:`f77multiple.f`.

.. ifconfig:: calcephapi in ('Python')

    The example in Python language is :file:`pymultiple.py`.

.. ifconfig:: calcephapi in ('Mex')

    The example in Octave/Matlab language is :file:`mexmultiple.m`.
