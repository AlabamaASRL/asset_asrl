This function begins the sequential writting of a type 14 segment (Chebyshev polynomials of the positions and velocities, with variable timesteps) to the SPK file associated to the ephemeris descriptor *eph*.
Once this function is called, no other segment can be written to the same file, and only |writeph_spk14_add| can be used with *eph*, until |writeph_spk14_end| is called.

.. include:: examples/writeph_spk14_begin.rst