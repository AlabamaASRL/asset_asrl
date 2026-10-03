This function opens an existing SPK ephemeris file in order to append ephemeris data to it, whose pathname is the string pointed to by *filename*,
and returns an ephemeris descriptor associated to it.
If the file doesn't exist, or if it isn't a SPK file, the function fails and returns NULL.
This file must be compliant to the format specified by the 'SPICE' SPK ephemeris file.

The function |writeph_close| must be called to flush data to the disk and to free memory allocated by this function.

.. include:: examples/writeph_spk_open.rst