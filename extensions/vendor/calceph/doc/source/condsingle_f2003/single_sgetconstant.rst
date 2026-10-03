
::

       integer res
       real(8) AU
       ! open the ephemeris file 
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
           if (calceph_sgetconstant("AU"//C_NULL_CHAR, AU).eq.1) then
               write (*,*) "AU=", AU
           endif
       endif
       
