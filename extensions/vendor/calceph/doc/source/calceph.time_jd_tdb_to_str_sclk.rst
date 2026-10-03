This function computes the spacecraft clock string corresponding to a given Barycentric Dynamical Time (TDB).

This is the inverse operation of |calceph_time_str_spacecraft_clock_to_jd_tdb|. The input Julian Date is provided as two double-precision numbers (*jd0_tdb*  and *jdfrac_tdb*). 

.. ifconfig:: calcephapi in ('C',)

    The resulting spacecraft clock string is stored in the character array *str*.

.. ifconfig:: calcephapi in ('F90', 'F2003')

     The resulting spacecraft clock string is stored in the character array *str*. Trailing blanks are added to the character array *str*.

It is strictly required to load a **Spacecraft Clock kernel** (usually ``.tsc``) to define the clock partitions and coefficients.

Since this function performs an internal conversion between Barycentric Dynamical Time (TDB) and Terrestrial Time (TT), additional kernels are required depending on the configuration set by |calceph_time_set_relationship_tt_tdb|:

* If the relationship mode is **0** (default), an **Ephemeris kernel** (usually ``.bsp``) containing the ntime tranformation TT-TDB must be loaded.
* If the relationship mode is **1**, a **Leap Seconds kernel** (usually ``.tls``) must be loaded.


If the time falls outside the range covered by the SCLK coefficients, then the function returns an error.

The following example converts a TDB date to a spacecraft clock string:

.. include:: examples/time_jd_tdb_to_str_sclk.rst
