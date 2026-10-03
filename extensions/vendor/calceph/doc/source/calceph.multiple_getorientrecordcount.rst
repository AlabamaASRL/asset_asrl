This function returns the number of orientation's records available in the ephemeris file |eph|. 
Usually, the number of records is equal to the number of bodies in the ephemeris file if the timespan is continuous. 
If the timespan is discontinuous for the target body, 
then each different timespan is counted as a different record.


The following example prints the number of orientation's records available in the ephemeris file

.. include:: examples/multiple_getorientrecordcount.rst
