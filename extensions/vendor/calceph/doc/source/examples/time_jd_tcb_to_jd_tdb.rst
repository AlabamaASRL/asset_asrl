.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tcb = 2451545.0;
        double jdfrac_tcb = 0.5;
        double jd0_tdb, jdfrac_tdb;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Convert TCB to TDB */
            if (calceph_time_jd_tcb_to_jd_tdb(peph, jd0_tcb, jdfrac_tcb, 
                                             &jd0_tdb, &jdfrac_tdb) != 0)
            {
                printf("TDB Date: %f + %f\n", jd0_tdb, jdfrac_tdb);
            }

            calceph_close(peph);
        }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
        integer res
        real(8) jd0, jdfrac
        TYPE(C_PTR) :: peph


        peph = calceph_open("example_lsk.tls"//C_NULL_CHAR)
        if (C_ASSOCIATED(peph)) then
            
            ! convert the julian day 2450083.1268436964601278 TCB to the julian day TDB
            res = calceph_time_jd_tcb_to_jd_tdb(peph, 450083.0D0, 0.1268436964601278D0, jd0, jdfrac)
            write(*,*) jd0, jdfrac ! print 22450083.0  0.1267361110076308

            call calceph_close(peph)
        endif

.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0, jdfrac
           
           res = f90calceph_open(peph, "example_lsk.tls")
           if (res.eq.1) then

             ! convert the julian day 2450083.1268436964601278 TCB to the julian day TDB
             res = f90calceph_time_jd_tcb_to_jd_tdb(peph, 450083.0D0, 0.1268436964601278D0, jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 222450083.0  0.1267361110076308

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        # convert the julian day 2450083.1268436964601278 TCB to the julian day TDB
        jd0, jdfrac = peph.time_jd_tcb_to_jd_tdb(2450083.0, 0.1268436964601278)
        print(jd0, jdfrac) # print 22450083.0  0.1267361110076308
        peph.close()



.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        # convert the julian day 2450083.126843696460127 TCB to the julian day TDB
        [jd0, jdfrac] = peph.time_jd_tcb_to_jd_tdb(2450083.0, 0.1268436964601278);
        printf("%f %.16f\n", jd0, jdfrac) # print 22450083.000000  0.1267361110076308
        peph.close()                