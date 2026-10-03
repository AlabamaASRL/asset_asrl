.. @c %----------------------------------------------------------------------------

.. include:: ../replace.rst

Single file access functions
============================

.. warning::   
    .. deprecated:: 4.0 
        Use :ref:`Reading/Evaluation functions` instead.

This group of functions works on a single ephemeris file at a given instant. 
They use an internal global variable to store information about the current opened ephemeris file. 

They are provided to have a similar interface of the fortran PLEPH function, supplied with the JPL ephemeris files. 
So the following call to PLEPH


.. code-block:: fortran
    
        PLEPH(46550D0, 3, 12, PV)
        

could be replaced by

.. code-block:: fortran
        
        calceph_sopen("ephemerisfile.dat")
        calceph_scompute(46550D0, 0, 3, 12, PV)
        calceph_sclose()


While the function PLEPH could access only one file in a program, these functions could access on multiple files in a program but not at same time. To access multiple files at a same time, the functions listed in the section :ref:`Reading/Evaluation functions`  must be used.  

When an error occurs, these functions execute error handlers according to the behavior defined by the function |calceph_seterrorhandler|. 


Time notes
----------

The function |calceph_scompute| only accepts a date expressed in the same timescale as the ephemeris files, which can be retrieved using the function |calceph_sgettimescale|. Ephemeris files are generally expressed using the timescale TDB.
If a date, expressed in the TT (Terrestrial Time) timescale, is supplied to this function, |calceph_scompute| will return an erroneous position of the order of several tens of meters for the planets.
If a date, expressed in the Coordinated Universal Time (UTC), is supplied to this function, |calceph_scompute| will return a very large erroneous position over several thousand kilometers for the planets.

  
Thread notes
------------

If the standard I/O functions such as *fread* are not reentrant
then the |LIBRARYSHORTNAME| I/O functions using them will not be reentrant either.

If the library was configured with the option *-DENABLE_THREAD=ON*,
these functions use an internal global variable per thread. Each thread could access to different ephemeris file and compute ephemeris data at same time. But each thread must call the function |calceph_sopen| to open ephemeris file even if all threads work on the same file.

If the library was configured with the default option *-DENABLE_THREAD=OFF*,
these functions use an internal global variable per process and are not thread-safe. If multiple threads are used in the process and call the function |calceph_scompute| at the same time, the caller thread must surround the call to this function with locking primitives, such as *pthread_lock/pthread_unlock* if POSIX Pthreads are used. 


Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions. 

The example in C language is :file:`csingle.c`. 
    
.. include:: single_prog.rst


Functions
---------

|menu_calceph_sopen|
~~~~~~~~~~~~~~~~~~~~

    
.. c:function:: int calceph_sopen ( const char *filename)

    :param filename: |arg_filename|
    :return: |retfuncfails0|


This function opens the file whose pathname is the string pointed to by filename, reads the header of this file
and associates an ephemeris descriptor to an internal variable. 
This file must be an ephemeris file.

This file must be compliant to the format specified by the  'original JPL binary' , 'INPOP 2.0 binary' or 'SPICE' ephemeris file. 
At the moment, supported SPICE files are the following :


 * text Planetary Constants Kernel (KPL/PCK) files
 * binary PCK  (DAF/PCK) files.
 * binary SPK (DAF/SPK) files containing segments of type |supportedspk|.
 * meta kernel (KPL/MK) files.
 * frame kernel (KPL/FK) files. Only a basic support is provided.

The function |calceph_sclose| must be called to free allocated memory by this function.

The following example opens the ephemeris file example1.dat 

::

    int res;
    res = calceph_sopen("example1.dat");
    if (res)
    {
        /* 
         ...  computation ... 
        */
        calceph_sclose();
    }

.. %------------------------------------------------

|menu_calceph_scompute|
~~~~~~~~~~~~~~~~~~~~~~~


.. c:function:: int calceph_scompute ( double JD0, double time, int target, int center, double PV[6] )

    :param  JD0: |arg_JD0|
    :param  time: |arg_time|
    :param  target: |arg_target|
    :param  center: |arg_center|
    :param  PV:  .. include:: ../arg_PV.rst
    :return: |retfuncfails0|

This function reads, if needed, and interpolates a single object, usually the position and velocity of one body (*target*) relative to another (*center*), from the ephemeris file, previously opened with the function |calceph_sopen|, for the time *JD0+time* and stores the results to *PV*. 


The date (JD0, time) should be expressed in the same timescale as the ephemeris files, which can be retrieved using the function |calceph_sgettimescale|.  

.. warning::
    If a date, expressed in the Coordinated Universal Time (UTC), is supplied to this function, a very large erroneous position will be returned. 


To get the best precision for the interpolation, the time is splitted in two floating-point numbers. The argument *JD0* should be an integer and *time* should be a fraction of the day. But you may call this function with *time=0* and *JD0*, the desired time, if you don't take care about precision.


The possible values for *target* and *center* are  :

+--------------------------------------+-------------------------+
| value                                |            meaning      |
+======================================+=========================+
| 1                                    | Mercury Barycenter      |
+--------------------------------------+-------------------------+
| 2                                    | Venus Barycenter        |
+--------------------------------------+-------------------------+
| 3                                    | Earth                   |
+--------------------------------------+-------------------------+
| 4                                    | Mars Barycenter         |
+--------------------------------------+-------------------------+
| 5                                    | Jupiter Barycenter      |
+--------------------------------------+-------------------------+
| 6                                    | Saturn Barycenter       |
+--------------------------------------+-------------------------+
| 7                                    | Uranus Barycenter       |
+--------------------------------------+-------------------------+
| 8                                    | Neptune Barycenter      |
+--------------------------------------+-------------------------+
| 9                                    | Pluto Barycenter        |
+--------------------------------------+-------------------------+
| 10                                   | Moon                    |
+--------------------------------------+-------------------------+
| 11                                   | Sun                     |
+--------------------------------------+-------------------------+
| 12                                   | Solar Sytem barycenter  |
+--------------------------------------+-------------------------+
| 13                                   | Earth-moon barycenter   |
+--------------------------------------+-------------------------+
| 14                                   | Nutation angles         |
+--------------------------------------+-------------------------+
| 15                                   | Librations              |
+--------------------------------------+-------------------------+
| 16                                   | TT-TDB                  |
+--------------------------------------+-------------------------+
| 17                                   | TCG-TCB                 |
+--------------------------------------+-------------------------+
| asteroid number + CALCEPH_ASTEROID   | asteroid (2E6+...)      |
+--------------------------------------+-------------------------+
| asteroid number + CALCEPH_ASTEROID_8 | asteroid (2E8+...)      |
+--------------------------------------+-------------------------+

These accepted values by this function are the same as the value for the JPL function *PLEPH*, except for the values *TT-TDB*, *TCG-TCB* and asteroids.

For example, the value "CALCEPH_ASTEROID+4" for target or center specifies the asteroid Vesta.

The following example prints the heliocentric coordinates of Mars at time=2451624.5 and at 2451624.9 

.. include :: single_scompute.rst


.. %------------------------------------------------

|menu_calceph_sgetconstant|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_sgetconstant ( const char* name, double *value )

    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfails0|


This function returns the value associated to the constant *name* in the header of the ephemeris file.

Only the first value is returned if multiple values are associated to a constant, such as a list of values.

The function |calceph_sopen| must be previously called  before.

The following example prints the value of the astronomical unit stored in the ephemeris file

.. include :: single_sgetconstant.rst


.. %------------------------------------------------

|menu_calceph_sgetconstantcount|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_sgetconstantcount ( )

    :return: |retfuncfails0|


This function returns the number of constants available in the header of the ephemeris file.

The function |calceph_sopen| must be previously called  before.

The following example prints the number of available constants stored in the ephemeris file

.. include :: single_sgetconstantcount.rst


.. %------------------------------------------------

|menu_calceph_sgetconstantindex|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. c:function:: int calceph_sgetconstantindex ( int index, char name[CALCEPH_MAX_CONSTANTNAME], double *value)

    :param  index: |arg_constant_index|
    :param  name: |arg_constant_name|
    :param  value: |arg_constant_value|
    :return: |retfuncfails0|



This function returns the name and its value of the constant available at the specified index in the header of the ephemeris file. The value of *index* must be between 1 and |calceph_sgetconstantcount|.


The function |calceph_sopen| must be previously called before.


The following example displays the name of the constants, stored in the ephemeris file, and their values 

.. include :: single_sgetconstantindex.rst

.. %------------------------------------------------

|menu_calceph_sgetfileversion|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


.. c:function:: int calceph_sgetfileversion ( char version[CALCEPH_MAX_CONSTANTVALUE])

    :param  version: |arg_fileversion|
    :return: |retfuncnotfound0|


This function returns the version of the ephemeris file, as a string. For example, the argument version will contain 'INPOP10B', 'EPM2017' or 'DE405', ... . 

If the file is an original JPL binary planetary ephemeris, then the version of the file can always be determined.
If the file is a spice kernel, the version of the file is retrieved from the constant *INPOP_PCK_VERSION*, *EPM_PCK_VERSION*, or *PCK_VERSION*.

The function |calceph_sopen| must be previously called before.

The following example prints the version of the ephemeris file. 

.. include:: single_sgetfileversion.rst

.. %------------------------------------------------

|menu_calceph_sgettimescale|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


.. c:function:: int calceph_sgettimescale ()

    :return: |retfuncfails0|


This function returns the timescale of the ephemeris file : 
  * 1 if the quantities of all bodies are expressed in the TDB time scale. 
  * 2 if the quantities of all bodies are expressed in the TCB time scale. 
  
The function |calceph_sopen| must be previously called before.

The following example prints the time scale available in the ephemeris file

.. include:: single_sgettimescale.rst

.. %------------------------------------------------

|menu_calceph_sgettimespan|
~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

.. c:function:: int calceph_sgettimespan (double* firsttime, double* lasttime, int* continuous )

    :param  firsttime: |arg_firsttime|
    :param  lasttime: |arg_lasttime|
    :param  continuous: |arg_continuous|
    :return: |retfuncfails0|


This function returns the first and last time available in the ephemeris file. The Julian date for the first and last time are expressed in the time scale returned by  |calceph_sgettimescale|
. 

It returns the following value in the parameter *continuous* :

  * 1 if the quantities of all bodies are available for any time between the first and last time. 
  * 2 if the quantities of some bodies are available on discontinuous time intervals between the first and last time. 
  * 3 if the quantities of each body are available on a continuous time interval between the first and last time, but not available for any time between the first and last time. 
  
The function |calceph_sopen| must be previously called before.

The following example prints the first and last time available in the ephemeris file

.. include:: single_sgettimespan.rst

.. %------------------------------------------------

|menu_calceph_sclose|
~~~~~~~~~~~~~~~~~~~~~

.. c:function:: void calceph_sclose ( )
    

This function closes the ephemeris data file and frees allocated memory by the function |calceph_sopen|.

