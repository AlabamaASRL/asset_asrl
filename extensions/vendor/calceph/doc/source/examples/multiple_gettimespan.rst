.. ifconfig:: calcephapi in ('C')

    ::

         double firsttime, lasttime;
         int countinuous;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           if (calceph_gettimespan(peph,  &firsttime, &lasttime, &countinuous)) 
               printf("%23.16E %23.16E %d\n", firsttime, lasttime, countinuous);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           integer :: continuous
           real(8) :: firsttime, lasttime
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
                if (calceph_gettimespan(peph, firsttime, lasttime, continuous).eq.1) then
                   write (*,*) firsttime, lasttime, countinuous
               endif

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           integer continuous
           double precision firsttime, lasttime
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
              if (f90calceph_gettimespan(peph, firsttime, lasttime, continuous).eq.1) then
                   write (*,*) firsttime, lasttime, countinuous
               endif
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        firsttime, lasttime, continuous = peph.gettimespan()
        print(firsttime, lasttime, continuous)
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        [firsttime, lasttime, continuous ] = peph.gettimespan()
        peph.close();

