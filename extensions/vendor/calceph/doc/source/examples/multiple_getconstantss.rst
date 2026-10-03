.. ifconfig:: calcephapi in ('C')

    ::

         t_calcephcharvalue unit;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* print the value of UNIT */
           if (calceph_getconstantss(peph, "UNIT", unit)) 
               printf("UNIT=%s\n", unit);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) UNIT
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
               ! print the value of UNIT 
               if (calceph_getconstantss(peph, "UNIT"//C_NULL_CHAR, UNIT).eq.1) then
                   write (*,*) "UNIT=", trim(UNIT)
               endif

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE) UNIT
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
              ! print the value of UNIT 
              if (f90calceph_getconstantss(peph, "UNIT", UNIT).eq.1) then
                   write (*,*) "UNIT=", trim(UNIT)
               endif
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        UNIT = peph.getconstantss("UNIT")
        print(UNIT)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        UNIT = peph.getconstantss('UNIT')
        peph.close();

