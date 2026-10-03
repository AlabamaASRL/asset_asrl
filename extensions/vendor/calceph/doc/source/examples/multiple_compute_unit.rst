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
           /* the heliocentric coordinates of Mars in km and km/s */
           calceph_compute_unit(peph, jd0, dt1, 4, 11,
                                CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, 
                                PV);
           for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);
   
           /* compute same quantity as the previous call using NAIF ID */
           calceph_compute_unit(peph, jd0, dt1, 
                                NAIFID_MARS_BARYCENTER, 
                                NAIFID_SUN,
                                CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM
                                +CALCEPH_UNIT_SEC, 
                                PV);
           for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);

          /* the heliocentric coordinates of Mars in AU and AU/day */
           calceph_compute_unit(peph, jd0, dt1, 4, 11, 
                                CALCEPH_UNIT_AU+CALCEPH_UNIT_DAY, 
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
 
              ! the heliocentric coordinates of Mars  in km and km/s
              res = calceph_compute_unit(peph,jd0, dt1, 4, 11, 
           &                            CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC,
           &                            PV)
              write(*,*) PV
              
              ! compute same quantity as the previous call using NAIF ID
              res = calceph_compute_unit(peph,jd0, dt1, 
           &                            NAIFID_MARS_BARYCENTER, NAIFID_SUN, 
           &                            CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC,
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

             ! the heliocentric coordinates of Mars  in km and km/s
             res = f90calceph_compute_unit(peph,jd0, dt1, 4, 11,
           &                               CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC,
           &                               PV)
             write(*,*) PV
             
             ! compute same quantity as the previous call using NAIF ID
             res = f90calceph_compute_unit(peph,jd0, dt1,
           &                               NAIFID_MARS_BARYCENTER, NAIFID_SUN, 
           &                               CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC,
           &                               PV)
             write(*,*) PV

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        def printcoord(PV,name):
            print('{0} :\n{1}\n'.format(name,PV))

        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open("example1.dat")

        PV1 = peph.compute_unit(jd0, dt, 4, 11, Constants.UNIT_KM+Constants.UNIT_SEC)
        printcoord(PV1,"heliocentric coordinates of Mars")

        PV2 = peph.compute_unit(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
        printcoord(PV2,"heliocentric coordinates of Mars")
    
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open('example1.dat');

        disp('heliocentric coordinates of Mars')
        PV1 = peph.compute_unit(jd0, dt, 4, 11, Constants.UNIT_KM+Constants.UNIT_SEC)
        

        disp('heliocentric coordinates of Mars')
        PV2 = peph.compute_unit(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
    
        peph.close();

