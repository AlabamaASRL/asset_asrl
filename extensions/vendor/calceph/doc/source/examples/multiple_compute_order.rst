.. ifconfig:: calcephapi in ('C')

    ::

         int j;
         double jd0=2442457;
         double dt1=0.5E0;
         t_calcephbin *peph;
         double PVAJ[12];
         double P[3];
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           /* compute only the heliocentric position of Mars in km */
           calceph_compute_order(peph, jd0, dt1,
                                NAIFID_MARS_BARYCENTER, 
                                NAIFID_SUN,
                                CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM
                                +CALCEPH_UNIT_SEC, 
                                0, P);
           for(j=0; j<3; j++) printf("%23.16E\n", P[j]);
   
           /* compute positions, velocities, accelerations and jerks 
           of Mars in km and seconds */
           calceph_compute_order(peph, jd0, dt1, 
                                NAIFID_MARS_BARYCENTER, 
                                NAIFID_SUN,
                                CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM
                                +CALCEPH_UNIT_SEC, 
                                3, PVAJ);
           for(j=0; j<12; j++) printf("%23.16E\n", PVAJ[j]);
 
           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           real(8) jd0
           real(8) dt1
           real(8) P(3)
           real(8) PVAJ(12)
           TYPE(C_PTR) :: peph
           
           jd0 = 2442457
           dt1 = 0.5D0
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
 
              ! compute only the heliocentric position of Mars in km
              res = calceph_compute_order(peph, jd0, dt1,
           &                     NAIFID_MARS_BARYCENTER, 
           &                     NAIFID_SUN,
           &                     CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, 
           &                     0, P);
              write(*,*) P
              
              ! compute positions, velocities, accelerations and jerks of Mars in km and seconds
              res = calceph_compute_order(peph, jd0, dt1, 
           &                     NAIFID_MARS_BARYCENTER, 
           &                     NAIFID_SUN,
           &                     CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, 
           &                     3, PVAJ);
              write(*,*) PVAJ

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0
           double precision dt1
           double precision P(3), PVAJ(12)
           
           jd0 = 2442457
           dt1 = 0.5D0
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then

             ! compute only the heliocentric position of Mars in km
             res = f90calceph_compute_order(peph, jd0, dt1,
           &                     NAIFID_MARS_BARYCENTER, 
           &                     NAIFID_SUN,
           &                     CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, 
           &                     0, P);
             write(*,*) P
             
             ! compute positions, velocities, accelerations and jerks of Mars in km and seconds
             res = f90calceph_compute_order(peph, jd0, dt1, 
           &                     NAIFID_MARS_BARYCENTER, 
           &                     NAIFID_SUN,
           &                     CALCEPH_USE_NAIFID+CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, 
           &                     3, PVAJ);
             write(*,*) PVAJ

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open("example1.dat")

        # compute only the heliocentric position of Mars in km
        P = peph.compute_order(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID, 0)
        print(P)

        # compute positions, velocities, accelerations and jerks of Mars in km and seconds
        PVAJ = peph.compute_order(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID, 3)
        print(PVAJ)
    
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        jd0=2442457
        dt=0.5E0
        
        peph = CalcephBin.open('example1.dat');

        % compute only the heliocentric position of Mars in km
        P = peph.compute_order(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID, 0)

        % compute positions, velocities, accelerations and jerks of Mars in km and seconds
        PVAJ = peph.compute_order(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                                Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID, 3)
    
        peph.close();


