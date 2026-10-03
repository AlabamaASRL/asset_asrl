This function creates for writing a new PCK file whose pathname is the string pointed to by *filename*,
and returns an ephemeris descriptor associated to it.
If a file with the same name already exists, it is overwritten.

The function |writeph_close| must be called to free memory allocated by this function.

The function |writeph_dump| and |writeph_restore| can be called to create a checkpoint restart if the creation of the file is very long.

.. include:: examples/writeph_pck_create.rst