.. ifconfig:: calcephapi in ('C')

    ::

         int nvalue, k;
         double *radii;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* get the number of values */
           int nvalue = calceph_getconstantvd(peph, "BODY399_RADII", NULL, 0);
           radii = (double*)malloc(sizeof(double)*nvalue);
           
           /* fill  the array radii */
           if (calceph_getconstantvd(peph, "BODY399_RADII", radii, nvalue))
           { 
               for (k=0; k<nvalue; k++)
               printf("BODY399_RADII(%d)=%23.16E\n", k, radii[k]);
           } 
           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res, nvalue
           real(8) svalue
           real(8), allocatable :: radii
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
                ! get the number of values 
                nvalue = calceph_getconstantsd(peph, "BODY399_RADII"//C_NULL_CHAR, svalue)
                ! fill the array
                allocate(radii(1:nvalue))
                res = calceph_getconstantvd(peph, "BODY399_RADII"//C_NULL_CHAR, radii, nvalue)
                write(*,*) radii

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res, nvalue
           double precision, allocatable :: radii
           double precision svalue
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
                ! get the number of values 
                nvalue = calceph_getconstantsd(peph, "BODY399_RADII", svalue)
                ! fill the array
                allocate(radii(1:nvalue))
                res = calceph_getconstantvd(peph, "BODY399_RADII", radii, nvalue)
                write(*,*) radii
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        radii = peph.getconstantvd("BODY399_RADII")
        print(radii)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        radii = peph.getconstantvd('BODY399_RADII')
        peph.close();

