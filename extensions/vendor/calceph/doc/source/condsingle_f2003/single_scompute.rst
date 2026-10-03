::

       integer res
       real(8) jd0
       real(8) dt1, dt2
       real(8) PV(6)

       jd0 = 2442457
       dt1 = 0.5D0
       dt2 = 0.9D0
       
       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then
            ! the heliocentric coordinates of Mars 
            res = calceph_scompute(jd0, dt1, 4, 11, PV)
            write (*,*) PV

            res = calceph_scompute(jd0, dt2, 4, 11, PV)
            write (*,*) PV
      
            call calceph_sclose
       endif     
