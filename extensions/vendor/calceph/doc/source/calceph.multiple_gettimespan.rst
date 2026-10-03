This function returns the first and last time available in the ephemeris file |eph|. The Julian date for the first and last time are expressed in the time scale returned by  |calceph_gettimescale|
. 

It returns the following value in the parameter *continuous* :

  * 1 if the quantities of all bodies are available for any time between the first and last time. 
  * 2 if the quantities of some bodies are available on discontinuous time intervals between the first and last time. 
  * 3 if the quantities of each body are available on a continuous time interval between the first and last time, but not available for any time between the first and last time. 
  

The following example prints the first and last time available in the ephemeris file

.. include:: examples/multiple_gettimespan.rst
