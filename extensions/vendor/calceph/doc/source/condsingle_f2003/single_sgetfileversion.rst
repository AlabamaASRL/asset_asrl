::

       integer res
       character(len=CALCEPH_MAX_CONSTANTVALUE) version
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
       
         res = calceph_sgetfileversion(version)
         write (*,*) "The version of the file is ", version
         
         call calceph_sclose
       endif
  