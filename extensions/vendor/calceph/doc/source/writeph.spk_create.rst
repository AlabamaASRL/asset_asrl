This function creates for writing a new SPK file whose pathname is the string pointed to by *filename*,
and returns an ephemeris descriptor associated to it.
If a file with the same name already exists, it is overwritten.

The function |writeph_close| must be called to free memory allocated by this function.

.. include:: examples/writeph_spk_create.rst