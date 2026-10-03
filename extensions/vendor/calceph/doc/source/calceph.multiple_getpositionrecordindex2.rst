
This function returns the target and origin bodies, the first and last time, the reference frame, and the segment type available at the specified index for the position's records of the ephemeris file |eph|. 
The NAIF identification numbering system is used for the target and center integers (:ref:`NAIF identification numbers` for the list).
The Julian date for the first and last time are expressed in the time scale returned by |calceph_gettimescale|. 

It returns the following value in the parameter *frame* :

+--------+-----------+
| value  | Name      |
+========+===========+
|  1     | ICRF      | 
+--------+-----------+
  
.. ifconfig:: calcephapi in ('C', 'F2003', 'F90')
    
    It returns in the parameter *segid* one of the predefined constants *CALCEPH_SEGTYPE_...* (:ref:`Constants`).

.. ifconfig:: calcephapi in ('Python', 'Mex')
    
    It returns in the parameter *segid* one of the predefined constants *Constants.SEGTYPE_...* (:ref:`Constants`).


The following example displays the position's records stored in the ephemeris file.

.. include:: examples/multiple_getpositionrecordindex2.rst
