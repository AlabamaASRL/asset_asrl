.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb, jdfrac_tdb;
        int target = 28; 
        const char *str = "1/0707109102:11026"; 
        const char *filenames[] = {"example_sclk.tsc", "example_lsk.tls"};

        /* open the ephemeris files */
        peph = calceph_open_array(2, filenames);
        calceph_time_set_relationship_tt_tdb(peph, 1);

        if (peph != NULL)
        {
            /* Convert SCLK string to TDB */
            if (calceph_time_str_spacecraft_clock_to_jd_tdb(peph, target, str,
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
        character(len=256), dimension (2) :: filear
        filear(1) = "example_lsk.tls"//C_NULL_CHAR
        filear(2) = "example_sclk.tsc"//C_NULL_CHAR
        peph = calceph_open_array(2, filear, 256) 
        if (C_ASSOCIATED(peph)) then
            
            res = calceph_time_set_relationship_tt_tdb(peph, 1)
            res = calceph_time_str_spacecraft_clock_to_jd_tdb(peph, 28, "1/0707109102:11026"//C_NULL_CHAR, jd0, jdfrac)
            write(*,*) jd0, jdfrac 

            call calceph_close(peph)
        endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           double precision jd0, jdfrac
           character*80 filear(2)

           filear(1) = "example_lsk.tls"
           filear(2) = "example_sclk.tsc"
           res = f90calceph_open_array(peph, 2, filear, 80)
           if (res.eq.1) then

             res = f90calceph_time_set_relationship_tt_tdb(peph, 1)
             res = f90calceph_time_str_spacecraft_clock_to_jd_tdb(peph, 28, "1/0707109102:11026", jd0, jdfrac)
             write(*,*) jd0, jdfrac 

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open(['example_lsk.tls', 'example_sclk.tsc'])
        peph.time_set_relationship_tt_tdb(1)
        jd0, jdfrac = peph.time_str_spacecraft_clock_to_jd_tdb(28, "1/0707109102:11026")
        print(jd0, jdfrac) # print 2459728.0 0.63395731523633
        peph.close()



.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open(cellstr({'example_lsk.tls', 'example_sclk.tsc'}))
        peph.time_set_relationship_tt_tdb(1);
        [jd0, jdfrac] = peph.time_str_spacecraft_clock_to_jd_tdb(28, "1/0707109102:11026");
        printf("%f %.16f\n", jd0, jdfrac) # print 2459728.000000 0.6339573152363300
        peph.close();                      