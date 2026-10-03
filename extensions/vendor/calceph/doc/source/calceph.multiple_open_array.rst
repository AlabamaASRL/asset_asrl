This function opens n files whose pathnames are the string pointed to by array_filename, 
reads the header blocks of these files
and returns a single ephemeris descriptor associated to them.  

.. ifconfig:: calcephapi in ('Mex')

    The array of files must be a cell array of character vectors (see details about cellstr) , and not an string or character arrays.

These files must have the same type (e.g., all files are SPICE files or original JPL files). 
This file must be compliant to the format specified by the  'original JPL binary' , 'INPOP 2.0 or 3.0 binary' or 'SPICE' ephemeris file. 
At the moment, supported SPICE files are the following :

 * text Planetary Constants Kernel (KPL/PCK) files
 * binary PCK  (DAF/PCK) files.
 * binary SPK (DAF/SPK) files containing segments of type |supportedspk|.
 * meta kernel (KPL/MK) files.
 * frame kernel (KPL/FK) files. Only a basic support is provided.



The single descriptor internally maintains a table to respond to the following queries for computing the position-velocity vector between bodies present in these different files.


Just after the call of |calceph_open_array|, the function |calceph_prefetch| should be called to accelerate future computations.

The function |calceph_close| must be called to free allocated memory by this function.



The following example opens the ephemeris file example1.bsp and example1.tpc

.. include:: examples/multiple_open_array.rst