.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;

        /* open the ephemeris file */
        peph = calceph_open("example_lsk.tls");
        if (peph != NULL)
        {
            /* Force using the default model based on TLS file for TT-TDB */
            if (calceph_time_set_relationship_tt_tdb(peph, 1) != 0)
            {
                printf("Relationship set to model 1.\n");
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
            res = calceph_time_jd_tt_to_jd_tdb(peph, 2460679.0, 0.4590218056322423, jd0, jdfrac)

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
             res = f90calceph_time_jd_tt_to_jd_tdb(peph, 2460679.0D0, 0.4590218056322423D0, jd0, jdfrac)

             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1) 
        jd0, jdfrac = peph.time_jd_tt_to_jd_tdb(2457754.0, 0.5)
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_lsk.tls")
        peph.time_set_relationship_tt_tdb(1);
        [jd0, jdfrac] = peph.time_jd_tt_to_jd_tdb(2460679.0, 0.4590218056322423);
        peph.close()  