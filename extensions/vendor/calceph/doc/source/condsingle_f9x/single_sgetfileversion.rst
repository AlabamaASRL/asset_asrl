
::

       integer res
       character(len=CALCEPH_MAX_CONSTANTVALUE) version
       ! open the ephemeris file 
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
       
         res = f90calceph_sgetfileversion(version)
         write (*,*) "The version of the file is ", version

         call f90calceph_sclose
       endif
      
