.. ifconfig:: calcephapi in ('C')

    ::

         int moon;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
            /* print the id of the Moon using the old numbering system */
            if (calceph_getidbyname(peph, "Moon", 0, &moon)) 
               printf("Moon=%d\n", moon);
            /*  print the id of the Moon  using the NAIF identification numbering system */
            if (calceph_getidbyname(peph, "Moon", CALCEPH_USE_NAIFID, &moon)) 
               printf("Moon=%d\n", moon);

            /* close the ephemeris file */
            calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           integer moon
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
               ! print the id of the Moon using the old numbering system
               if (calceph_getidbyname(peph, "Moon"//C_NULL_CHAR, 0, moon).eq.1) then
                   write (*,*) "Moon=", moon
               endif
               ! print the id of the Moon  using the NAIF identification numbering system
               if (calceph_getidbyname(peph, "Moon"//C_NULL_CHAR, CALCEPH_USE_NAIFID, moon).eq.1) then
                   write (*,*) "Moon=", moon
               endif
               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           integer moon
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
              ! print the id of the Moon using the old numbering system
              if (f90calceph_getidbyname(peph, "Moon", 0, moon).eq.1) then
                   write (*,*) "Moon=", moon
               endif
              ! print the id of the Moon  using the NAIF identification numbering system
              if (f90calceph_getidbyname(peph, "Moon", CALCEPH_USE_NAIFID, moon).eq.1) then
                   write (*,*) "Moon=", moon
              endif 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        # print the id of the Moon using the old numbering system
        moon = peph.getidbyname('Moon', 0)
        print(moon)
        # print the id of the Moon using the NAIF identification numbering system
        moon = peph.getidbyname('Moon', Constants.USE_NAIFID)
        print(moon)
        print(NaifId.MOON)
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        # print the id of the Moon using the old numbering system
        moon = peph.getidbyname('Moon', 0)
        print(moon)
        # print the id of the Moon using the NAIF identification numbering system
        moon = peph.getidbyname('Moon', Constants.USE_NAIFID)
        print(moon)
        print(NaifId.MOON)
        peph.close();

