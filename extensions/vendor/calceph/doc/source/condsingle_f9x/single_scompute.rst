::

       double precision PV(6)
       integer res
       double precision jd0
       double precision dt1, dt2

       jd0 = 2442457
       dt  = 0.5D0
       dt2 = 0.9D0
       
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then

          ! the heliocentric coordinates of Mars 
          res = f90calceph_scompute(jd0, dt1, 4, 11, PV)
          write (*,*) PV

          res = f90calceph_scompute(jd0, dt2, 4, 11, PV)
          write (*,*) PV

         call f90calceph_sclose
       endif  
