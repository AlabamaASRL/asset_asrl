.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0 = 2451545.0; /* J2000.0 */
        double jdfrac = 0.5;    /* 12h */
        int yy, mm, dd, hh, min;
        double sec;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* convert JD to Calendar (UTC) */
            if (calceph_time_jd_to_cal(peph, CALCEPH_UTC, jd0, jdfrac, 
                                       &yy, &mm, &dd, &hh, &min, &sec) != 0)
            {
                printf("Date: %d-%d-%d %d:%d:%f\n", yy, mm, dd, hh, min, sec);
            }

            /* close the file */
            calceph_close(peph);
        }

.. ifconfig:: calcephapi in ('F2003')

    ::
    
        integer res, yy, mo, dd, hh,mi
        real(8) ss
        TYPE(C_PTR) :: peph


        peph = calceph_open("example_lsk.tls"//C_NULL_CHAR)
        if (C_ASSOCIATED(peph)) then
            
            res = calceph_time_set_relationship_tt_tdb(peph, 1)
            ! convert the julian day 2457754.499994212962963 UTC to the calendar date UTC
            res = calceph_time_jd_to_cal(peph, 2460679.0, 0.45902180578559637, yy, mo, dd, hh,mi, ss)
            write(*,*) yy, mo, dd, hh,mi, ss ! print 2016 12 31 23 59 60.49999421295723 (leap second)

            call calceph_close(peph)
        endif

.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res, yy, mo, dd, hh,mi
           double precision ss
           
           res = f90calceph_open(peph, "example_lsk.tls")
           if (res.eq.1) then

             res = f90calceph_time_set_relationship_tt_tdb(peph, 1)
             ! convert the julian day 2457754.499994212962963 UTC (leap second) to the calendar date UTC
             res = f90calceph_time_jd_to_cal(peph, 2460679.0D0, 0.45902180578559637D0, yy, mo, dd, hh,mi, ss)
             write(*,*) yy, mo, dd, hh,mi, ss ! print 2016 12 31 23 59 60.49999421295723 

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1)
        # convert the julian day 2457754.499994212962963 UTC (leap second) to the calendar date UTC
        yy, mo, dd, hh,mi, ss = peph.time_jd_to_cal(Constants.UTC, 2457754, 0.499994212962963)
        print(yy, mo, dd, hh,mi, ss) # print 2016 12 31 23 59 60.49999421295723 
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        # convert the julian day 2457754.499994212962963 UTC (leap second) to the calendar date UTC
        [yy, mo, dd, hh,mi, ss] = peph.time_jd_to_cal(Constants.UTC, 2457754, 0.499994212962963);
        printf("%d-%d-%d %d:%d:%.6f\n", yy, mo, dd, hh,mi, ss) # print 2016-12-31 23:59:60.499994 
        peph.close()        