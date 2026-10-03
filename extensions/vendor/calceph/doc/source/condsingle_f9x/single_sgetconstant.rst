

::

       integer res
       double precision AU

       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
           if (f90calceph_sgetconstant("AU", AU).eq.1) then
               write (*,*) "AU=", AU
           endif
        endif
  
