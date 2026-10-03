
This function returns 1 if multiple threads can access the same ephemeris ephemeris descriptor, otherwise 0. 

A previous call to the function |calceph_prefetch| is required, and the library should be compiled with **--enable-thread=yes** on Unix-like operating system,  for the function |calceph_isthreadsafe| to return a non-zero value. If the file is not encoded with the same endian as the current hardware, then function may return 0.

If this function returns 1, several threads may use the same ephemeris descriptor for the computational functions |calceph_compute|, .... It allows to use the same object for the parallel loops. 
