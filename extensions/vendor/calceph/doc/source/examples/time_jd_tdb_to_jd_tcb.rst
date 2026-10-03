.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb = 2451545.0;
        double jdfrac_tdb = 0.5;
        double jd0_tcb, jdfrac_tcb;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Convert TDB to TCB */
            if (calceph_time_jd_tdb_to_jd_tcb(peph, jd0_tdb, jdfrac_tdb, 
                                             &jd0_tcb, &jdfrac_tcb) != 0)
            {
                printf("TCB Date: %f + %f\n", jd0_tcb, jdfrac_tcb);
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
            
            ! convert the julian day 2450083.1267361110076308 TDB to the julian day TCB
            res = calceph_time_jd_tdb_to_jd_tcb(peph,2450083.0, 0.1267361110076308D0, jd0, jdfrac)
            write(*,*) jd0, jdfrac ! print 2450083.0 0.1268436964601278

            call calceph_close(peph)
        endif

.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0, jdfrac
           
           res = f90calceph_open(peph, "example_lsk.tls")
           if (res.eq.1) then

             ! convert the julian day 2450083.1267361110076308 TDB to the julian day TCB
             res = f90calceph_time_jd_tdb_to_jd_tcb(peph, 2450083.0D0, 0.1267361110076308D0, jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 2450083.0 0.1268436964601278

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        # convert the julian day 2450083.1267361110076308 TDB to the julian day TCB
        jd0, jdfrac = peph.time_jd_tdb_to_jd_tcb(2450083.0, 0.1267361110076308)
        print(jd0, jdfrac) # print 2450083.0 0.1268436964601278
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        # convert the julian day 2450083.1267361110076308 TDB to the julian day TCB
        [jd0, jdfrac] = peph.time_jd_tdb_to_jd_tcb(2450083.0, 0.1267361110076308);
        printf("%f %.16f\n", jd0, jdfrac) # print  2450083.000000 0.1268436964601278
        peph.close()        