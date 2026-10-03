

::

       integer res
       integer n
       ! open the ephemeris file 
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
       
           n = f90calceph_sgetconstantcount()
           write (*,*) "number of constants", n

           call f90calceph_sclose
       endif
  
