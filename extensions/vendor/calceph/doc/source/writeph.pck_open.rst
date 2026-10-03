This function opens an existing PCK ephemeris file in order to append ephemeris data to it, whose pathname is the string pointed to by *filename*,
and returns an ephemeris descriptor associated to it.
If the file doesn't exist, or if it isn't a PCK file, the function fails and returns NULL.
This file must be compliant to the format specified by the 'SPICE' PCK ephemeris file.

The function |writeph_close| must be called to free memory allocated by this function.

.. include:: examples/writeph_pck_open.rst