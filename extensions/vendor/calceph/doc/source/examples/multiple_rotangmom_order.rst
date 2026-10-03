.. ifconfig:: calcephapi in ('C')

    ::

         int j;
         double jd0=2451419;
         double dt1=0.5E0;
         t_calcephbin *peph;
         double G[3];
 
         /* open the ephemeris file */
         peph = calceph_open("example2_rotangmom.dat");
         if (peph)
         {
           calceph_prefetch(peph);
   
           calceph_rotangmom_order(peph, jd0, dt1, NAIFID_EARTH,
                   CALCEPH_USE_NAIFID+CALCEPH_UNIT_SEC, 
                   0,
                   G);
           for(j=0; j<3; j++) printf("%23.16E\n", G[j]);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           real(8) jd0
           real(8) dt1
           real(8) G(3)
           TYPE(C_PTR) :: peph
           
           jd0 = 2451419
           dt1 = 0.5D0
           peph = calceph_open("example2_rotangmom.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
 
              res = calceph_rotangmom_order(peph,jd0, dt1, NAIFID_EARTH, 
           &                            CALCEPH_USE_NAIFID+CALCEPH_UNIT_SEC,
           &                            G)
              write(*,*) G

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0
           double precision dt1
           double precision G(3)
           
           jd0 = 2451419
           dt1 = 0.5D0
           res = f90calceph_open(peph, "example2_rotangmom.dat")
           if (res.eq.1) then

             res = f90calceph_rotangmom_order(peph,jd0, dt1, NAIFID_EARTH,
           &                              CALCEPH_USE_NAIFID+CALCEPH_UNIT_SEC,
           &                              G)
             write(*,*) G

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        jd0=2451419
        dt=0.5E0
        
        peph = CalcephBin.open("example2_rotangmom.dat")

        G = peph.rotangmom_order(jd0, dt, NaifId.EARTH,
                              Constants.USE_NAIFID+Constants.UNIT_SEC, 0)
        print(G)
    
        peph.close()



.. ifconfig:: calcephapi in ('Mex')

    ::
    
        jd0=2451419
        dt=0.5E0
        
        peph = CalcephBin.open('example2_rotangmom.dat')

        G = peph.rotangmom_order(jd0, dt, NaifId.EARTH,
                              Constants.USE_NAIFID+Constants.UNIT_SEC, 0)
    
        peph.close()


