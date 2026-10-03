.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb = 2455348.5;
        double jdfrac_tdb = 0.0;
        int yy, mm, dd, hh, min;
        double sec;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Direct conversion: TDB JD -> UTC Calendar */
            if (calceph_time_jd_tdb_to_cal_utc(peph, jd0_tdb, jdfrac_tdb, 
                                               &yy, &mm, &dd, &hh, &min, &sec) != 0)
            {
                printf("UTC Date: %d-%d-%d %d:%d:%f\n", yy, mm, dd, hh, min, sec);
            }

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
            ! convert 2025-01-03T22:59:50.300 UTC to the julian day TDB
            res = calceph_time_jd_tdb_to_cal_utc(peph, 2460679.0, 0.45902180578559637, yy, mo, dd, hh,mi, ss)
            write(*,*) yy, mo, dd, hh,mi, ss ! print 2025 1 3 22 59 50.30000662574196

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
             ! convert the julian day 2460679.45902180578559637 TDB to the calendar date UTC
             res = f90calceph_time_jd_tdb_to_cal_utc(peph, 2460679.0D0, 0.45902180578559637D0, yy, mo, dd, hh,mi, ss)
             write(*,*) yy, mo, dd, hh,mi, ss ! print 2025 1 3 22 59 50.30000662574196

             call f90calceph_close(peph)
           endif



.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1)
        # convert the julian day 2460679.45902180578559637 TDB to the calendar date UTC
        yy, mo, dd, hh,mi, ss = peph.time_jd_tdb_to_cal_utc(2460679.0, 0.45902180578559637)
        print(yy, mo, dd, hh,mi, ss) # print 2025 1 3 22 59 50.30000662574196
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        # convert the julian day 2460679.45902180578559637 TDB to the calendar date UTC
        [yy, mo, dd, hh,mi, ss] = peph.time_jd_tdb_to_cal_utc(2460679.0, 0.45902180578559637);
        printf("%d-%d-%d %d:%d:%.6f\n", yy, mo, dd, hh,mi, ss) # print  2025-1-3 22:59:50.300007
        peph.close()        