
.. @c %-------------------------------------------------------------------------

.. include:: replace.rst

.. _`SPK writing functions`:

Writing functions
=================

These functions are used to create and write data to binary SPK and PCK ephemerids files.


|LIBRARYNAME| is able to write the following segment type into the SPK files.

+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Data representation                                                      | Time spacing      | Timescale | Segment type | Sequential functions                      | Parallel functions                                      |
+==========================================================================+===================+===========+==============+===========================================+=========================================================+
| Chebyshev polynomials of the positions                                   | uniform block     | TDB       | 2            | writeph_spk2_seq_write                    | writeph_spk2_par_reserve and writeph_spk2_par_write     |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the positions                                   | uniform block     | TCB       | 102          | writeph_spk102_seq_write                  | writeph_spk102_par_reserve and writeph_spk102_par_write |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the positions and velocities                    | uniform block     | TDB       | 3            | writeph_spk3_seq_write                    | writeph_spk3_par_reserve and writeph_spk3_par_write     |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the positions and velocities                    | uniform block     | TCB       | 103          | writeph_spk103_seq_write                  | writeph_spk103_par_reserve and writeph_spk103_par_write |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the positions and velocities                    | non-uniform block | TDB       | 14           | writeph_spk14_seq_begin, ..._add, ..._end |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| discrete states of the positions and velocities                          | equal timesteps   | TDB       | 8            | writeph_spk8_seq_write                    | writeph_spk8_par_reserve and writeph_spk8_par_write     |
|                                                                          |                   |           |              |                                           |                                                         |
| (Lagrange interpolation)                                                 |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| discrete states of the positions and velocities                          | unequal timesteps | TDB       | 9            | writeph_spk9_seq_write                    | writeph_spk9_par_reserve and writeph_spk9_par_write     |
|                                                                          |                   |           |              |                                           |                                                         |
| (Lagrange interpolation)                                                 |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| discrete states of the positions and velocities                          | equal timesteps   | TDB       | 12           | writeph_spk12_seq_write                   | writeph_spk12_par_reserve and writeph_spk12_par_write   |
|                                                                          |                   |           |              |                                           |                                                         |
| (Hermite interpolation)                                                  |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| discrete states of the positions and velocities                          | unequal timesteps | TDB       | 13           | writeph_spk13_seq_write                   | writeph_spk13_par_reserve and writeph_spk13_par_write   |
|                                                                          |                   |           |              |                                           |                                                         |
| (Hermite interpolation)                                                  |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+




|LIBRARYNAME| is able to write the following segment type into the PCK files.

+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Data representation                                                      | Time spacing      | Timescale | Segment type | Sequential functions                      | Parallel functions                                      |
+==========================================================================+===================+===========+==============+===========================================+=========================================================+
| Chebyshev polynomials of the orientations                                | uniform block     | TDB       | 2            | writeph_pck2_seq_write                    | writeph_pck2_par_reserve and writeph_pck2_par_write     |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the orientations                                | uniform block     | TCB       | 102          | writeph_pck102_seq_write                  | writeph_pck102_par_reserve and writeph_pck102_par_write |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the orientations                                | uniform block     | TDB       | 3            | writeph_pck3_seq_write                    | writeph_pck3_par_reserve and writeph_pck3_par_write     |
|                                                                          |                   |           |              |                                           |                                                         |
| and their derivatives                                                    |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+
| Chebyshev polynomials of the orientations                                | uniform block     | TCB       | 103          | writeph_pck103_seq_write                  | writeph_pck103_par_reserve and writeph_pck103_par_write |
|                                                                          |                   |           |              |                                           |                                                         |
| and their derivatives                                                    |                   |           |              |                                           |                                                         |
+--------------------------------------------------------------------------+-------------------+-----------+--------------+-------------------------------------------+---------------------------------------------------------+


.. @c %-------------------------------------------------------------------------

Thread notes
------------

There are two modes to write SPK files : sequential and parallel.

In sequential mode, segments are written one after the other using either the function *writeph_spkN_seq_write*, 
or the fuctions *writeph_spkN_begin*, *writeph_spkN_add* and *writeph_spkN_end*, with N the type of segment.

Once *writeph_spkN_begin* is called, no other segment can be written to the file it has be called with, until *writeph_spkN_end* is called. (Even using *writeph_spkN_seq_write*.)
Hence *writeph_spkN_add* is the only writing function that can be used between *writeph_spkN_begin* and *writeph_spkN_end* for a same file.

In parallel mode, a segment can be written by several threads simultaneously using the functions *writeph_spkN_par_reserve* and *writeph_spkN_par_write*, with N the type of segment.
First, the function *writeph_spkN_par_reserve* is called to reserve space in the file for several segments.
Then, multiple threads can call simultaneously the function *writeph_spkN_par_write* to write records in the reserved segments, **it is the responsability of the user to ensure that no two threads write the same record of a segment**.

NB: The parallel mode is only efficient if the underlying file system supports concurrent write access to the same file from multiple threads and multithreading libraries (e.g. pthreads). Otherwise the performance may be worse than the sequential mode.


Usage
-----

The following examples, that can be found in the directory *examples* of the library sources, show the typical usage of this group of functions.

.. ifconfig:: calcephapi in ('C')

    The example in C language is :file:`cwriteph.c`. 


.. ifconfig:: calcephapi in ('F2003')

    The example in Fortran 2003 language is :file:`f2003writeph.f`.

.. @c %-------------------------------------------------------------------------

SPK Functions
-------------

.. toctree::
    :maxdepth: 4

    writeph.spk.cusage.rst
    writeph.spk.f2003usage.rst

.. @c %-------------------------------------------------------------------------

PCK Functions
-------------

.. toctree::
    :maxdepth: 4

    writeph.pck.cusage.rst
    writeph.pck.f2003usage.rst