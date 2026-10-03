.. ifconfig:: calcephapi in ('C')

    ::

         double AU;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* print the value of AU */
           if (calceph_getconstantsd(peph, "AU", &AU)) 
               printf("AU=%23.16E\n", AU);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           real(8) AU
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
                ! print the value of AU 
                if (calceph_getconstantsd(peph, "AU"//C_NULL_CHAR, AU).eq.1) then
                   write (*,*) "AU=", AU
               endif

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision AU
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
              ! print the value of AU 
              if (f90calceph_getconstantsd(peph, "AU", AU).eq.1) then
                   write (*,*) "AU=", AU
               endif
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        AU = peph.getconstantsd("AU")
        print(AU)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        AU = peph.getconstantsd('AU')
        peph.close();

