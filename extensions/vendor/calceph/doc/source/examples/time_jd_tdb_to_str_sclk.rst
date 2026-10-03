.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        double jd0_tdb = 2459728.0;
        double jdfrac_tdb = 0.63395731542095745681;
        int target = 28;
        t_calcephcharvalue str;
        const char *filenames[] = {"example_sclk.tsc", "example_lsk.tls"};

        /* open the ephemeris files */
        peph = calceph_open_array(2, filenames);

        calceph_time_set_relationship_tt_tdb(peph, 1);

        if (peph != NULL)
        {
            /* Convert TDB to SCLK string */
            if (calceph_time_jd_tdb_to_str_spacecraft_clock(peph, target,
                                                            jd0_tdb, jdfrac_tdb, str) != 0)
            {
                printf("SCLK string: %s\n", str);
            }

            calceph_close(peph);
        }



.. ifconfig:: calcephapi in ('F2003')

    ::
    
        integer res
        TYPE(C_PTR) :: peph
        character(len=CALCEPH_MAX_CONSTANTVALUE) strdate
        character(len=256), dimension (2) :: filear

        filear(1) = "example_lsk.tls"//C_NULL_CHAR
        filear(2) = "example_sclk.tsc"//C_NULL_CHAR
        peph = calceph_open_array(2, filear, 256) 
        if (C_ASSOCIATED(peph)) then
            
            res = calceph_time_set_relationship_tt_tdb(peph, 1)
            res = calceph_time_jd_tdb_to_str_spacecraft_clock(peph, 28, 2459728D0, 0.63395731542095745681D0, strdate)
            write(*,*) strdate 

            call calceph_close(peph)
        endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE) strdate
           character*80 filear(2)

           filear(1) = "example_lsk.tls"
           filear(2) = "example_sclk.tsc"
           res = f90calceph_open_array(peph, 2, filear, 80)
           if (res.eq.1) then

             res = f90calceph_time_set_relationship_tt_tdb(peph, 1)
             res = f90calceph_time_jd_tdb_to_str_spacecraft_clock(peph, 28, 2459728D0, 0.63395731542095745681D0, strdate)
             write(*,*) strdate

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open(['example_lsk.tls', 'example_sclk.tsc'])
        peph.time_set_relationship_tt_tdb(1)
        strdate = peph.time_jd_tdb_to_str_spacecraft_clock(28, 2459728, 0.63395731542095745681)
        print(strdate) # print "1/0707109102:11026"
        peph.close()



.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open(cellstr({'example_lsk.tls', 'example_sclk.tsc'}))
        peph.time_set_relationship_tt_tdb(1);
        strdate = peph.time_jd_tdb_to_str_spacecraft_clock(28, 2459728, 0.63395731542095745681);
        printf("%s\n", strdate) # print "1/0707109102:11026"
        peph.close()                        