.. ifconfig:: calcephapi in ('C')

    ::
    
     int res;
     int j;
     double jd0=2442457;
     double dt1=0.5E0;
     double dt2=0.9E0;
     t_calcephbin *peph;
     double PV[6];
 
     /* open the ephemeris file */
     peph = calceph_open("example1.dat");
     if (peph)
     {
       /* the heliocentric coordinates of Mars */
       calceph_compute(peph, jd0, dt1, 4, 11, PV);
       for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);
   
       calceph_compute(peph, jd0, dt2, 4, 11, PV);
       for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);
 
       /* close the ephemeris file */
       calceph_close(peph);
     }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer*8 peph
           integer res
           real(8) jd0
           real(8) dt1, dt2
           real(8) PV(6)
           TYPE(C_PTR) :: peph
           
           jd0 = 2442457
           dt1 = 0.5D0
           dt2 = 0.9D0
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
 
              ! the heliocentric coordinates of Mars 
              res = calceph_compute(peph,jd0, dt1, 4, 11, PV)
              write(*,*) PV
              
              res = calceph_compute(peph,jd0, dt2, 4, 11, PV)
              write(*,*) PV

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer res
           double precision jd0
           double precision dt1, dt2
           double precision PV(6)
           
           jd0 = 2442457
           dt1 = 0.5D0
           dt2 = 0.9D0
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then

             res = f90calceph_compute(peph,jd0, dt1, 4, 11, PV)
             write(*,*) PV
             
             res = f90calceph_compute(peph,jd0, dt2, 4, 11, PV)
             write(*,*) PV

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        def printcoord(PV,name):
            print('{0} :\n{1}\n'.format(name,PV))

        jd0=2442457
        dt1=0.5E0
        dt2=0.9E0
        
        peph = CalcephBin.open("example1.dat")

        PV1 = peph.compute(jd0, dt1, 4, 11)
        printcoord(PV1,"heliocentric coordinates of Mars")

        PV2 = peph.compute(jd0, dt2, 4, 11)
        printcoord(PV2,"heliocentric coordinates of Mars")
    
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        jd0=2442457
        dt1=0.5E0
        dt2=0.9E0
        
        peph = CalcephBin.open('example1.dat');

        disp('heliocentric coordinates of Mars')
        PV1 = peph.compute(jd0, dt1, 4, 11)

        disp('heliocentric coordinates of Mars')
        PV2 = peph.compute(jd0, dt2, 4, 11)
    
        peph.close();


