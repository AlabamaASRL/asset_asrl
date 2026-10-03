.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0, jdfrac;
        const char *str = "2000-01-01T12:00:00";

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Parse string as UTC and convert to the julian day UTC */
            if (calceph_time_str_to_jd(peph, CALCEPH_UTC, str, &jd0, &jdfrac) != 0)
            {
                printf("Julian Date from string: %f + %f\n", jd0, jdfrac);
            }

            /* close the file */
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
            ! convert the calendar date TAI to the julian day TAI
            res = calceph_time_str_to_jd(peph, CALCEPH_TAI, "2025-01-03T22:59:50.300"//C_NULL_CHAR, jd0, jdfrac)
            write(*,*) jd0, jdfrac ! print 2460679.0 0.45822106481481484

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
             ! convert the calendar date TAI to the julian day TAI
             res = f90calceph_time_str_to_jd(peph, CALCEPH_TAI, "2025-01-03T22:59:50.300", jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 2460679.0 0.45822106481481484

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1)
        # convert the calendar date TAI to the julian day TAI
        jd0, jdfrac = peph.time_str_to_jd(Constants.TAI, "2025-01-03T22:59:50.300")
        print(jd0, jdfrac) # print 2460679.0 0.45822106481481484
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        # convert the calendar date TAI to the julian day TAI
        [jd0, jdfrac] = peph.time_str_to_jd(Constants.TAI, "2025-01-03T22:59:50.300");
        printf("%f %.16f\n", jd0, jdfrac) # print 2460679.000000 0.4582210648148148
        peph.close()                        