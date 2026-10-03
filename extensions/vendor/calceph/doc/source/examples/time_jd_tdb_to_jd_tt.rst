.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb = 2451545.0;
        double jdfrac_tdb = 0.5;
        double jd0_tt, jdfrac_tt;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Convert TDB to TT */
            if (calceph_time_jd_tdb_to_jd_tt(peph, jd0_tdb, jdfrac_tdb, 
                                             &jd0_tt, &jdfrac_tt) != 0)
            {
                printf("TT Date: %f + %f\n", jd0_tt, jdfrac_tt);
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
            
            res = calceph_time_set_relationship_tt_tdb(peph, 1)
            ! convert the julian day 2460679.45902180578559637 TDB to the julian day TT
            res = calceph_time_jd_tdb_to_jd_tt(peph, 2460679.0, 0.45902180578559637, jd0, jdfrac)
            write(*,*) jd0, jdfrac ! print 2460679.0 0.4590218056322423

            call calceph_close(peph)
        endif

.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0, jdfrac
           
           res = f90calceph_open(peph, "example_lsk.tls")
           if (res.eq.1) then

             res = f90calceph_time_set_relationship_tt_tdb(peph, 1)
             ! convert the julian day 2460679.45902180578559637 TDB to the julian day TT
             res = f90calceph_time_jd_tdb_to_jd_tt(peph, 2460679.0D0, 0.45902180578559637D0, jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 2460679.0 0.4590218056322423

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1)
        # convert the julian day 2460679.45902180578559637 TDB to the julian day TT
        jd0, jdfrac = peph.time_jd_tdb_to_jd_tt(2460679.0, 0.45902180578559637)
        print(jd0, jdfrac) # print 2460679.0 0.4590218056322423
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        # convert the julian day 2460679.45902180578559637 TDB to the julian day TT
        [jd0, jdfrac] = peph.time_jd_tdb_to_jd_tt(2460679.0, 0.45902180578559637);
        printf("%f %.16f\n", jd0, jdfrac) # print  2460679.000000 0.4590218056322423
        peph.close()        