.. ifconfig:: calcephapi in ('C')

    ::

     int count;
     t_calcephbin *peph;
 
     /* open the ephemeris file */
     peph = calceph_open("example1.dat");
     if (peph)
     {
       /* print the number of constants */
       count = calceph_getconstantcount(peph);
       printf("number of constants : %d\n", count);

       /* close the ephemeris file */
       calceph_close(peph);
     }

.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer n
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
             n = calceph_getconstantcount(peph)
             write (*,*) "number of constants", n
             call calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           integer n
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
             n = f90calceph_getconstantcount(peph)
             write (*,*) "number of constants", n
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        n = peph.getconstantcount()
        print("number of constants", n)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        n = peph.getconstantcount()
        printf('number of constants=%d\n', n)
        peph.close();
