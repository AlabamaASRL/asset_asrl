This function returns the target body, the first and last time, and the reference frame available at the specified index for the orientation's records of the ephemeris file |eph|. 
The NAIF identification numbering system is used for the target body (:ref:`NAIF identification numbers` for the list).
The Julian date for the first and last time are expressed in the time scale returned by |calceph_gettimescale|. 

It returns the following value in the parameter *frame* :

+--------+-----------+
| value  | Name      |
+========+===========+
|  1     | ICRF      | 
+--------+-----------+
  

The following example displays the orientation's records stored in the ephemeris file.

.. include:: examples/multiple_getorientrecordindex.rst