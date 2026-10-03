::

       integer res
       integer cont
       real(8) jdfirst, jdlast
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
       
         res = calceph_sgettimespan(jdfirst, jdlast, cont)
         write (*,*) "data available between ", jdfirst, "and", jdlast
         write (*,*) "continuous data ", cont
         
         call calceph_sclose
       endif
