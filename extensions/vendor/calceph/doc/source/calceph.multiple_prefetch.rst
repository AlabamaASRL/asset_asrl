
This function prefetches to the main memory all files associated to |ephemerisdescriptoreph|. 
This prefetching operation will accelerate the further computations performed with |calceph_compute|, |calceph_compute_unit|,  |calceph_compute_order|, |calceph_orient_unit|, ... . 

It requires that the file is smaller than the main memory.
If multiple threads (e.g. threads of openMP or Posix Pthreads) prefetch the data for the same ephemeris file, 
the used memory will remain the same as if the prefetch operation was done by a single thread if and if the 
endianess of the file is the same as the computer and if the operating system, such as Linux, MacOS X other unix, supports the function mmap.
