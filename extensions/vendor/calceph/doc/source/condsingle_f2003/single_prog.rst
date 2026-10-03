::

     USE, INTRINSIC :: ISO_C_BINDING
     use calceph
     implicit none
     integer res
     real(8) AU, EMRAT, GM_Mer
     real(8) jd0
     real(8) dt
     real(8) PV(6)
     integer j
     real(8) valueconstant
     character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant

     jd0 = 2442457
     dt = 0.5E0
     ! open the ephemeris file 
     res = calceph_sopen("example1.dat"//C_NULL_CHAR)
     if (res.eq.1) then
       write (*,*) "The ephemeris is already opened"
       ! print the values of AU, EMRAT and GM_Mer 
       if (calceph_sgetconstant("AU"//C_NULL_CHAR, AU).eq.1) then
           write (*,*) "AU=", AU
       endif
       if (calceph_sgetconstant("EMRAT"//C_NULL_CHAR,EMRAT).eq.1) then
           write (*,*) "EMRAT=", EMRAT
       endif
       if (calceph_sgetconstant("GM_Mer"//C_NULL_CHAR,GM_Mer).eq.1) then
           write (*,*) "GM_Mer=", GM_Mer
       endif

       ! compute and print the coordinates 
       ! the geocentric moon coordinates 
       res = calceph_scompute(jd0, dt, 10, 3, PV)
       call printcoord(PV,"geocentric coordinates of the Moon")
       ! the value TT-TDB 
       if (calceph_scompute(jd0, dt, 16, 0, PV).eq.1) then
         write (*,*) "TT-TDB = ", PV(1)
       endif
       ! the heliocentric coordinates of Mars 
       res = calceph_scompute(jd0, dt, 4, 11, PV)
       call printcoord(PV,"heliocentric coordinates of Mars")


       ! print the whole list of the constants 
       write (*,*) "list of constants"
       do j=1, calceph_sgetconstantcount()
         res = calceph_sgetconstantindex(j,nameconstant, valueconstant)
         write (*,*) nameconstant,"=",valueconstant
       enddo


       ! close the ephemeris file 
       call calceph_sclose
       write (*,*) "The ephemeris is already closed"
     else
       write (*,*) "The ephemeris can't be opened"
     endif
       
