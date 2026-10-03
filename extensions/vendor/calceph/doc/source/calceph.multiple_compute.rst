.. ifconfig:: calcephapi in ('Mex')

    *JD0* and *time* could be arrays of double-precision floating-point values.

This function reads, if needed, in the ephemeris file |eph| and interpolates a single object, usually the position and velocity of one body (*target*) relative to another (*center*) for the time *JD0+time* and stores the results to *PV*. The ephemeris file |eph| must have been previously opened with the function |calceph_open|.

The returned array *PV* has the following properties

  - If the target is *TT-TDB*, only the first element of this array will get the result. The time scale transformation TT-TDB is expressed in seconds.
  - If the target is *TCG-TCB*, only the first element of this array will get the result. The time scale transformation TCG-TCB is expressed in seconds.
  - If the target is *Librations*, the array contains the angles of the librations of the Moon and their derivatives. The angles of the librations of the Moon are expressed in radians and their derivatives are expressed in radians per day.
  - If the target is *Nutations*, the array contains the nutation angles and their derivatives. The nutation angles are expressed in radians and their derivatives are expressed in radians per day.
  - Otherwise the returned values is the cartesian position (x,y,z), expressed in Astronomical Unit (au), and the velocity (xdot, ydot, zdot), expressed in Astronomical Unit per day (au/day). 

.. ifconfig:: calcephapi in ('Python')

   If *JD0* and *time* are list or NumPy's array (1D) of double-precision floating-point values, the returned array *PV* is a list of 6 arrays. Each array contain a single component of position or velocity (e.g., PV[0] contains the coordinate X, PV[1] contains the coordinate Y, ...) . 

|timescale_JD0_Time| To get the best numerical precision for the interpolation, the time is splitted in two floating-point numbers. The argument *JD0* should be an integer and *time* should be a fraction of the day. But you may call this function with *time=0* and *JD0*, the desired time, if you don't take care about numerical precision.

.. warning::
    |warning_UTC|


The possible values for *target* and *center* are  :

+--------------------------------------+-------------------------+
| value                                |            meaning      |
+======================================+=========================+
| 1                                    | Mercury Barycenter      |
+--------------------------------------+-------------------------+
| 2                                    | Venus Barycenter        |
+--------------------------------------+-------------------------+
| 3                                    | Earth                   |
+--------------------------------------+-------------------------+
| 4                                    | Mars Barycenter         |
+--------------------------------------+-------------------------+
| 5                                    | Jupiter Barycenter      |
+--------------------------------------+-------------------------+
| 6                                    | Saturn Barycenter       |
+--------------------------------------+-------------------------+
| 7                                    | Uranus Barycenter       |
+--------------------------------------+-------------------------+
| 8                                    | Neptune Barycenter      |
+--------------------------------------+-------------------------+
| 9                                    | Pluto Barycenter        |
+--------------------------------------+-------------------------+
| 10                                   | Moon                    |
+--------------------------------------+-------------------------+
| 11                                   | Sun                     |
+--------------------------------------+-------------------------+
| 12                                   | Solar Sytem barycenter  |
+--------------------------------------+-------------------------+
| 13                                   | Earth-moon barycenter   |
+--------------------------------------+-------------------------+
| 14                                   | Nutation angles         |
+--------------------------------------+-------------------------+
| 15                                   | Librations              |
+--------------------------------------+-------------------------+
| 16                                   | TT-TDB                  |
+--------------------------------------+-------------------------+
| 17                                   | TCG-TCB                 |
+--------------------------------------+-------------------------+
| asteroid number + CALCEPH_ASTEROID   | asteroid (2E6+...)      |
+--------------------------------------+-------------------------+
| asteroid number + CALCEPH_ASTEROID_8 | asteroid (2E8+...)      |
+--------------------------------------+-------------------------+

These accepted values by this function are the same as the value for the JPL function *PLEPH*, except for the values *TT-TDB*, *TCG-TCB* and asteroids.

For example, the value "CALCEPH_ASTEROID+4" for target or center specifies the asteroid Vesta.


The following example prints the heliocentric coordinates of Mars at time=2442457.5 and at 2442457.9 

.. include:: examples/multiple_compute.rst
    