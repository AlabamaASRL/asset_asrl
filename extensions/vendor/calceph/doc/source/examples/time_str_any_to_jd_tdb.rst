.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb, jdfrac_tdb;
        /* String with explicit timescale */
        const char *str = "2010-06-01T00:00:00 TAI";

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Parse string (detecting TAI) and convert to TDB */
            if (calceph_time_str_any_to_jd_tdb(peph, str, &jd0_tdb, &jdfrac_tdb) != 0)
            {
                printf("TDB Julian Date from TAI string: %f + %f\n", jd0_tdb, jdfrac_tdb);
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
            ! convert the calendar date UTC to the julian day TDB
            res = calceph_time_str_any_to_jd_tdb(peph, "2025-01-03T22:59:50.300 UTC"//C_NULL_CHAR, jd0, jdfrac)
            write(*,*) jd0, jdfrac ! print 2460679.0 0.45902180578559637

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
             ! convert the calendar date UTC to the julian day TDB
             res = f90calceph_time_str_any_to_jd_tdb(peph, "2025-01-03T22:59:50.300 UTC", jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 2460679.0 0.45902180578559637

             call f90calceph_close(peph)
           endif

.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1)
        # convert the calendar date UTC to the julian day TDB
        jd0, jdfrac = peph.time_str_any_to_jd_tdb("2025-01-03T22:59:50.300 UTC")
        print(jd0, jdfrac) # print 2460679.0 0.45902180578559637
        peph.close()



.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        # convert the calendar date UTC to the julian day TDB
        [jd0, jdfrac] = peph.time_str_any_to_jd_tdb("2025-01-03T22:59:50.300 UTC");
        printf("%f %.16f\n", jd0, jdfrac) # print 2460679.000000 0.4590218057855964
        peph.close()                        