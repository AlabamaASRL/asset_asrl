::

       integer res
       integer t
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
       
         t = calceph_sgettimescale()
         write (*,*) "timescale ", t
         
         call calceph_sclose
       endif
       
