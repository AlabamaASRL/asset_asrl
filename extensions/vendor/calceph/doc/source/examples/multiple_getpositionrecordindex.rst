.. ifconfig:: calcephapi in ('C')

    ::

     int j;
     int res;
     double firsttime, lasttime;
     int itarget, icenter, iframe;
     t_calcephbin *peph;
 
     /* open the ephemeris file */
     peph = calceph_open("example1.dat");
     if (peph)
     {
        for (j=1; j<=calceph_getpositionrecordcount(peph); j++)
        {
            calceph_getpositionrecordindex(peph, j, &itarget, &icenter, &firsttime, &lasttime, &iframe);
            printf("record %d : target=%d center=%d start=%23.16E end=%23.16E frame=%d\n", 
                   j, itarget, icenter,  firsttime, lasttime, iframe);
        }

        /* close the ephemeris file */
        calceph_close(peph);
     }

.. ifconfig:: calcephapi in ('F2003')

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           integer j, itarget, icenter, iframe
           real(C_DOUBLE) firsttime, lasttime
           TYPE(C_PTR) :: peph
           
        ! open the ephemeris file 
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then

        ! print the list of positionrecords 
             do j=1, calceph_getpositionrecordcount(peph)
               res = calceph_getpositionrecordindex(peph,j,itarget, icenter, firsttime, lasttime, iframe)
               write (*,*) itarget, icenter, firsttime, lasttime, iframe
             enddo

             call calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer j, itarget, icenter, iframe
           real*8 firsttime, lasttime
           integer res
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
             do j=1, f90calceph_getpositionrecordcount(peph)
               res = f90calceph_getpositionrecordindex(peph,j,itarget, icenter, firsttime, lasttime, iframe)
               write (*,*) itarget, icenter, firsttime, lasttime, iframe
             enddo
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        n = peph.getpositionrecordcount()
        for j in range(1, n+1):
            itarget, icenter, firsttime, lasttime, iframe = peph.getpositionrecordindex(j)
            print(itarget, icenter, firsttime, lasttime, iframe)
        
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        n = peph.getpositionrecordcount()
        for j=1:n
            [itarget, icenter, firsttime, lasttime, iframe] = peph.getpositionrecordindex(j)
        end
        peph.close();
