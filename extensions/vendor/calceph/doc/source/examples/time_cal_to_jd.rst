.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0, jdfrac;
        int yy = 2000, mm = 1, dd = 1;
        int hh = 12, min = 0;
        double sec = 0.0;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* convert Calendar (UTC) to JD */
            if (calceph_time_cal_to_jd(peph, CALCEPH_UTC, yy, mm, dd, hh, min, sec, 
                                       &jd0, &jdfrac) != 0)
            {
                printf("Julian Date: %f + %f\n", jd0, jdfrac);
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
            
            ! convert 2025-01-03T22:59:50.300 UTC to the julian day UTC
            res = calceph_time_cal_to_jd(peph, CALCEPH_UTC, 2025, 1, 3, 22, 59, 50.3, jd0, jdfrac)
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

             ! convert 2025-01-03T22:59:50.300 UTC to the julian day UTC
             res = f90calceph_time_cal_to_jd(peph, CALCEPH_UTC, 2025, 1, 3, 22, 59, 50.3D0, jd0, jdfrac)
             write(*,*) jd0, jdfrac ! print 2460679.0 0.45822106481481484

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        # convert 2025-01-03T22:59:50.300 UTC to the julian day UTC
        jd0, jdfrac = peph.time_cal_to_jd(Constants.UTC, 2025, 1, 3, 22, 59, 50.3)
        print(jd0, jdfrac) # print 2460679.0 0.45822106481481484
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        # convert 2025-01-03T22:59:50.300 UTC to the julian day UTC
        [jd0, jdfrac] = peph.time_cal_to_jd(Constants.UTC, 2025, 1, 3, 22, 59, 50.3);
        printf("%f %.16f\n", jd0, jdfrac) # print 2460679.000000 0.4582210648148148
        peph.close()