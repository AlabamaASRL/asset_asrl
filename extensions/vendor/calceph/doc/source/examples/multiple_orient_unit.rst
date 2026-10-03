.. ifconfig:: calcephapi in ('C')

    ::

         int j;
         double jd0=2442457;
         double dt1=0.5E0;
         t_calcephbin *peph;
         double PV[6];
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           calceph_prefetch(peph);
   
           calceph_orient_unit(peph, jd0, dt1, NAIFID_MOON,
                                CALCEPH_USE_NAIFID+CALCEPH_UNIT_RAD
                                +CALCEPH_UNIT_SEC, 
                                PV);
           for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           real(8) jd0
           real(8) dt1
           real(8) PV(6)
           TYPE(C_PTR) :: peph
           
           jd0 = 2442457
           dt1 = 0.5D0
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
 
              res = calceph_orient_unit(peph,jd0, dt1, NAIFID_MOON, 
           &                            CALCEPH_USE_NAIFID+CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC,
           &                            PV)
              write(*,*) PV

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0
           double precision dt1
           double precision PV(6)
           
           jd0 = 2442457
           dt1 = 0.5D0
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then

             res = f90calceph_orient_unit(peph,jd0, dt1, NAIFID_MOON,
           &                              CALCEPH_USE_NAIFID+CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC,
           &                              PV)
             write(*,*) PV

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open("example1.dat")

        PV = peph.orient_unit(jd0, dt, NaifId.MOON,
                              Constants.USE_NAIFID+Constants.UNIT_RAD+Constants.UNIT_SEC)
        print(PV)
    
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open('example1.dat');

        PV = peph.orient_unit(jd0, dt, NaifId.MOON,
                              Constants.USE_NAIFID+Constants.UNIT_RAD+Constants.UNIT_SEC)
        peph.close();

