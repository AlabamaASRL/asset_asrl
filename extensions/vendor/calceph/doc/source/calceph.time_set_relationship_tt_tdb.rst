This function sets the mathematical model or data source used to convert dates between the Terrestrial Time (TT) and Barycentric Dynamical Time (TDB) scales.

The selected model affects the behavior of the conversion functions |calceph_time_jd_tdb_to_jd_tt| and |calceph_time_jd_tt_to_jd_tdb|.

The supported values for the model argument are:

- 0: The conversion uses |calceph_compute_unit| to retrieve the time difference. The difference TT-TDB should be available in the ephemeris file. 

- 1: The conversion uses a default model based on the loaded TLS file.

If an invalid model is specified, the function returns an error.

The following example sets the relationship model to use the binary ephemeris data:

.. include:: examples/time_set_relationship_tt_tdb.rst
