
::

       integer res
       integer cont
       double precision  jdfirst, jdlast
       ! open the ephemeris file 
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
       
         res = f90calceph_sgettimespan(jdfirst, jdlast, cont)
         write (*,*) "data available between ", jdfirst, "and", jdlast
         write (*,*) "continuous data ", cont

           call f90calceph_sclose
       endif
      
