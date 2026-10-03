.. ifconfig:: calcephapi in ('C')

    ::

         int nvalue, k;
         t_calcephcharvalue *mission_units;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* get the number of values */
           int nvalue = calceph_getconstantvs(peph, "MISSION_UNITS", NULL, 0);
           mission_units = (t_calcephcharvalue*)malloc(sizeof(t_calcephcharvalue)*nvalue);
           
           /* fill  the array radii */
           if (calceph_getconstantvs(peph, "MISSION_UNITS", mission_units, nvalue))
           { 
               for (k=0; k<nvalue; k++)
               printf("MISSION_UNITS(%d)=%s\n", k, mission_units[k]);
           } 
           free(mission_units);
           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res, nvalue
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) svalue
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR), allocatable :: mission_units
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
                ! get the number of values 
                nvalue = calceph_getconstantss(peph, "MISSION_UNITS"//C_NULL_CHAR, svalue)
                ! fill the array
                allocate(mission_units(1:nvalue))
                res = calceph_getconstantvs(peph, "MISSION_UNITS"//C_NULL_CHAR, mission_units, nvalue)
                write(*,*) mission_units
                deallocate(mission_units)

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res, nvalue
           character(len=CALCEPH_MAX_CONSTANTVALUE), allocatable :: mission_units
           character(len=CALCEPH_MAX_CONSTANTVALUE) svalue
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
                ! get the number of values 
                nvalue = f90calceph_getconstantss(peph, "MISSION_UNITS", svalue)
                ! fill the array
                allocate(mission_units(1:nvalue))
                res = f90calceph_getconstantvs(peph, "MISSION_UNITS", mission_units, nvalue)
                write(*,*) mission_units
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        mission_units = peph.getconstantvs("MISSION_UNITS")
        print(mission_units)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        mission_units = peph.getconstantvs('MISSION_UNITS')
        peph.close();

