::

       integer res
       integer j
       real(8) valueconstant
       character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant

       res = calceph_sopen("example1.dat"//C_NULL_CHAR)
       if (res.eq.1) then

         do j=1, calceph_sgetconstantcount()
           res = calceph_sgetconstantindex(j,nameconstant,valueconstant)
           write (*,*) nameconstant,"=",valueconstant
         enddo

       call calceph_sclose
