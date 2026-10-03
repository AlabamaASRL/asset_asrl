
::

       integer res
       integer j
       double precision valueconstant
       character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant

       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
       
         do j=1, f90calceph_sgetconstantcount()
            res = f90calceph_sgetconstantindex(j,nameconstant,valueconstant)
            write (*,*) nameconstant,"=",valueconstant
         enddo
         
         call f90calceph_sclose
       endif

