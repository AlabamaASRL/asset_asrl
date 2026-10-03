This function writes a comment to the SPK file associated to the ephemeris descriptor *eph*.
The comment is a null-terminated string of characters pointed to by *comment*.
This function cannot be called after writing a segment to the SPK file (i.e. after calling any of the *writeph_spkN_seq_write* or *writeph_spkN_par_reserve* functions), otherwise it will fail.

.. include:: examples/writeph_spk_comment.rst