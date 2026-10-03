This function converts a spacecraft clock string (SCLK) to the Barycentric Dynamical Time (TDB) scale.

The input spacecraft clock string is provided in the parameter *str*. The converted date is stored in *jd0_tdb* and *jdfrac_tdb*.


It is strictly required to load a **Spacecraft Clock kernel** (usually ``.tsc``) to define the clock partitions and coefficients.

Since this function performs an internal conversion between Terrestrial Time (TT) and Barycentric Dynamical Time (TDB), additional kernels are required depending on the configuration set by |calceph_time_set_relationship_tt_tdb|:

* If the relationship mode is **0** (default), an **Ephemeris kernel** (usually ``.bsp``) containing the ntime tranformation TT-TDB must be loaded.
* If the relationship mode is **1**, a **Leap Seconds kernel** (usually ``.tls``) must be loaded.


If the required kernels are missing or if the string format is invalid, then the function returns an error.

The following example converts a spacecraft clock string to TDB:

.. include:: examples/time_str_sclk_to_jd_tdb.rst
