::

       integer res
       integer n
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
       
         n = calceph_sgetconstantcount()
         write (*,*) "number of constants", n
         
         call calceph_sclose
       endif
