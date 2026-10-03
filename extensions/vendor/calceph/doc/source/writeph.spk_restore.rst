This function restores the content of the ephemeris descriptor *eph* from a binary dump file whose pathname is the string pointed to by *filename*.
The dump file must have been created by the function |writeph_dump|.

This function can be called to restart from a checkpoint restart.

.. include:: examples/writeph_spk_dump.rst