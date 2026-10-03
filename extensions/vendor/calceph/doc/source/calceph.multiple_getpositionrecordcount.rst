
This function returns the number of position's records available in the ephemeris file |eph|. 
Usually, the number of records is equal to the number of bodies in the ephemeris file if the timespan is continuous. 
If the timespan is discontinuous for the target and center bodies, 
then each different timespan is counted as a different record.
If the ephemeris file contain time scale transformations' records, such as *TT-TDB* or *TCG-TCB*, then these records are included in the returned value.


The following example prints the number of position's records available in the ephemeris file

.. include:: examples/multiple_getpositionrecordcount.rst
