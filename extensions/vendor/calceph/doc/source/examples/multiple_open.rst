.. ifconfig:: calcephapi in ('C')

    The following example opens the ephemeris file example1.dat and example2.dat
    
    ::

         t_calcephbin *peph1;
         t_calcephbin *peph2;
         peph1 = calceph_open("example1.dat");
         peph2 = calceph_open("example2.dat");
         if (peph1 && peph2)
         {
           calceph_prefetch(peph1);
           calceph_prefetch(peph2);
           /* 
             ...  computation ... 
           */
         }
         /* close the files */
         if (peph1) calceph_close(peph1);
         if (peph2) calceph_close(peph2);

.. ifconfig:: calcephapi in ('F2003')

    The following example opens the ephemeris file example1.dat

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
           
                ! ... computation ... 
           
           endif 
           call calceph_close(peph)    

.. ifconfig:: calcephapi in ('F90')

    The following example opens the ephemeris file example1.dat

    ::

           include 'f90calceph.h'
           integer res

           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
           
                ! ... computation ... 
           
           endif 
           call f90calceph_close(peph)


.. ifconfig:: calcephapi in ('Python')

    The following example opens the ephemeris file example1.dat

    ::
    
        from calcephpy import *

        peph = CalcephBin.open("example1.dat")
 
        # ...  computation ...  

        peph.close()
 

.. ifconfig:: calcephapi in ('Mex')

    The following example opens the ephemeris file example1.dat

    ::
    
        peph = calceph.CalcephBin.open('example1.dat')
 
        % ...  computation ...  

        peph.close()
 
