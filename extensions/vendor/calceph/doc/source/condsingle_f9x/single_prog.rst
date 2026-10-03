::

   program f77single
       implicit none
       include 'f90calceph.h'
       integer res
       double precision AU, EMRAT, GM_Mer
       double precision jd0
       double precision dt
       double precision PV(6)
       integer j
       double precision valueconstant
       character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant

       jd0 = 2442457
       dt = 0.5E0
       ! open the ephemeris file 
       res = f90calceph_sopen("example1.dat")
       if (res.eq.1) then
           write (*,*) "The ephemeris is already opened"
           ! print the values of AU, EMRAT and GM_Mer 
           if (f90calceph_sgetconstant("AU", AU).eq.1) then
               write (*,*) "AU=", AU
           endif
           if (f90calceph_sgetconstant("EMRAT", EMRAT).eq.1) then
               write (*,*) "EMRAT=", EMRAT
           endif
           if (f90calceph_sgetconstant("GM_Mer", GM_Mer).eq.1)then
               write (*,*) "GM_Mer=", GM_Mer
           endif

          ! compute and print the coordinates 
          ! the geocentric moon coordinates 
          res = f90calceph_scompute(jd0, dt, 10, 3, PV)
          call printcoord(PV,"geocentric coordinates of the Moon")
          ! the value TT-TDB 
          if (f90calceph_scompute(jd0, dt, 16, 0, PV).eq.1) then
           write (*,*) "TT-TDB = ", PV(1)
          endif
          ! the heliocentric coordinates of Mars 
          res = f90calceph_scompute(jd0, dt, 4, 11, PV)
          call printcoord(PV,"heliocentric coordinates of Mars")

         ! print the whole list of the constants 
         write (*,*) "list of constants"
         do j=1, f90calceph_sgetconstantcount()
            res = f90calceph_sgetconstantindex(j,nameconstant,valueconstant)
            write (*,*) nameconstant,"=",valueconstant
         enddo

         ! close the ephemeris file 
         call f90calceph_sclose
         write (*,*) "The ephemeris is already closed"
        else
           write (*,*) "The ephemeris can't be opened"
        endif
   stop
   end
  
