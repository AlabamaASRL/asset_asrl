The following example creates a new ephemeris file ephemeris.bsp with segment of type 12 
for the heliocentric coordinates of Venus.

.. ifconfig:: calcephapi in ('C')

    ::

         t_writephbin *weph;
         const char segid[] = "seg_planet";

         weph = writeph_spk_create("ephemeris.bsp","planet_2", 0);
         if (weph)
         {
               int len_timespan = 32; /* days */
               int frame = 1; /* ICRF */
               int record_count = 100;
               int interpolation_degree = 7;
               double jd_start = 2460000;
               double jd_end = jd_start+record_count*len_timespan;
               double pos_vel[600];  /*  size = 100*6  = record_count* 6 components */

               for (int k=0; k<record_count; k++)
               {

                    double jd = jd_start+k*len_timespan;

                    /* ... fill the array pos_vel with the positions and velocities at the date jd ...
                         pos_vel[..] =...
                    */
               }
               writeph_spk12_seq_write(weph, NAIFID_VENUS, NAIFID_SUN, frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, pos_vel, record_count, interpolation_degree, segid);
               

               writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    ::

          USE, INTRINSIC :: ISO_C_BINDING
          use calceph
          TYPE(C_PTR) :: weph
          INTEGER len_timespan
          REAL(8) :: jd_start, jd_end, jd
          INTEGER frame, record_count, interpolation_degree, k, ret
          REAL(8), dimension(600) :: pos_vel !  size = 600 = record_count * 6 components
          
          len_timespan = 32 ! days 
          frame = 1 ! ICRF
          record_count = 100
          interpolation_degree = 7
          jd_start = 2460000
          jd_end = jd_start+record_count*len_timespan

          weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "planet_2"//C_NULL_CHAR,0)
          if (C_ASSOCIATED(weph)) then
          
               do k=0, record_count-1
                    jd = jd_start+k*len_timespan;
                    ! ... fill the array pos_vel with the positions and velocities at the date jd  ...
                    ! pos_vel(...) =...
               enddo
               ret = writeph_spk12_seq_write(weph, NAIFID_VENUS, NAIFID_SUN, frame, jd_start, 0.0, jd_end, 0.0, 
                              len_timespan, pos_vel, record_count, interpolation_degree, "seg_planet"//C_NULL_CHAR)
               
               call writeph_close(weph)    
          endif 

