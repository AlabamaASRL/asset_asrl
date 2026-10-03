

::

       integer res
       integer t
       ! open the ephemeris file 
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
       
           t = f90calceph_sgettimescale()
           write (*,*) "timescale ", t

           call f90calceph_sclose
       endif
  
