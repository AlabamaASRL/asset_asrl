
This function opens the file whose pathname is the string pointed to by filename, reads the two header blocks of this file
and returns an ephemeris descriptor associated to it. 
This file must be compliant to the format specified by the 'original JPL binary' , 'INPOP 2.0 binary' or 'SPICE' ephemeris file. 
At the moment, supported SPICE files are the following :

  * text Planetary Constants Kernel (KPL/PCK) files
  * binary PCK  (DAF/PCK) files.
  * binary SPK (DAF/SPK) files containing segments of type |supportedspk|.
  * meta kernel (KPL/MK) files.
  * frame kernel (KPL/FK) files. Only a basic support is provided.

Just after the call of |calceph_open|, the function |calceph_prefetch| should be called to accelerate future computations.

The function |calceph_close| must be called to free allocated memory by this function.

.. include:: examples/multiple_open.rst