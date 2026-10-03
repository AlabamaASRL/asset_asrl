This function dumps the content of the ephemeris descriptor *eph* to a binary dump file whose pathname is the string pointed to by *filename*.
The state of the ephemeris descriptor *eph* can be restored from this dump file using the function |writeph_restore|.

This function can be called to create a checkpoint restart if the creation of the file is very long.

.. include:: examples/writeph_spk_dump.rst