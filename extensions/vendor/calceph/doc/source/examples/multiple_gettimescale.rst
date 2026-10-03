.. ifconfig:: calcephapi in ('C')

    ::

         t_calcephbin *peph;
         int timescale;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* print the time scale */
           timescale = calceph_gettimescale(peph); 
           if (timescale==1) printf("timescale=TDB\n");
           if (timescale==2) printf("timescale=TCB\n");

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           integer timescale
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
              ! print the time scale 
              timescale = calceph_gettimescale(peph)
              write (*,*) "timescale=", timescale

              call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision AU
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
             ! print the time scale 
             timescale = calceph_gettimescale(peph)
             write (*,*) "timescale=", timescale
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        timescale = peph.gettimescale()
        print(timescale)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        timescale = peph.gettimescale()
        peph.close();

