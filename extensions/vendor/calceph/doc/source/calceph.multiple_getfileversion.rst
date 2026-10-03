This function returns the version of the ephemeris file, as a string. For example, the argument version will contain 'INPOP10B', 'EPM2017' or 'DE405', ... . 

If the file is an original JPL binary planetary ephemeris, then the version of the file can always be determined.
If the file is a spice kernel, the version of the file is retrieved from the constant *INPOP_PCK_VERSION*, *EPM_PCK_VERSION*, or *PCK_VERSION*.


The following example prints the version of the ephemeris file. 

.. include:: examples/multiple_getfileversion.rst