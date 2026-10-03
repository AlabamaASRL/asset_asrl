The following example creates a new ephemeris file ephemeris.bsp with segment of type 13 
for the heliocentric coordinates of Mercury and Venus using OpenMP.

.. ifconfig:: calcephapi in ('C')

    ::

         t_writephbin *weph;

         weph = writeph_spk_create("ephemeris.bsp","planet_2", 0);
         if (weph)
         {
               int frame = 1; /* ICRF */
               int target_count = 2;
               int targets[2] = { NAIFID_MERCURY, NAIFID_VENUS };
               int records_count[2] = { 10, 20 };
               int len_timespan[2] = { 32, 16 };
               const char segids[2] = { "seg_mercury",  "seg_venus" };
               double jd_start = 2460000;
               double jd_end = 2460000+32*10;
               int reservation;
               int interpolation_degree = 7;

               reservation = writeph_spk13_seq_reserve(weph, target_count, targets, NAIFID_SUN, 
                                      frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, record_counts, interpolation_degree, segids);

               #pragma omp parallel for
               for (int body = 0; body<target_count; body++)
               {
                    #pragma omp parallel for
                    for (int k=0; k<records_count[body]; k++)
                    {
                        double pos_vel[6]; /*  size = 6 components */
                        double epochs[1];  /*  size =  1 time */

                        epochs[0] = jd_start+k*len_timespan;

                        /* ... fill the array pos_vel with the positions and velocities at the date jd ...
                         pos_vel[..] =...
                        */
                        writeph_spk13_par_write(weph, reservation, body,  k, 1,  pos_vel, epochs);
                    }
               }
                

               writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    ::

          USE, INTRINSIC :: ISO_C_BINDING
          use calceph
          TYPE(C_PTR) :: weph
          REAL(8) :: jd_start, jd_end
          INTEGER frame, body, k, ret
          REAL(8), dimension(6) :: pos_vel !  size  = 6 components
          REAL(8), dimension(1) :: epochs !  size  = 1 date
          INTEGER, dimension(2) :: record_counts, targets
          INTEGER target_count, reservation
          REAL(8), dimension(2) :: len_timespan 
          CHARACTER(len=40), dimension (2) :: segids
          INTEGER interpolation_degree

          frame = 1 ! ICRF
          target_count = 2
          targets(1) = NAIFID_MERCURY
          targets(2) = NAIFID_VENUS
          record_counts(1) = 10
          record_counts(2) = 20
          segids(1) = "seg_mercury"//C_NULL_CHAR 
          segids(2) = "seg_venus"//C_NULL_CHAR 
          interpolation_degree = 7
          jd_start = 2460000
          jd_end = jd_start+10

          weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "planet_2"//C_NULL_CHAR,0)
          if (C_ASSOCIATED(weph)) then
          
               reservation = writeph_spk13_seq_reserve(weph, target_count, targets, NAIFID_SUN, 
                                      frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, record_counts, interpolation_degree, segids)
               
               !$omp parallel for private(body)
               do body=0, target_count
                    !$omp parallel for private(k, pos_vel)
                    do k=0, record_counts(body)-1
                        ! ... fill the array epochs and pos_vel with the positions and velocities  ...
                        ! epochs (0) = ...
                        ! pos_vel(...) =...
                        ret = writeph_spk13_par_write(weph, reservation, body,  k, 1,  pos_vel, epochs)
                    enddo
               enddo
               call writeph_close(weph)    
          endif 
       

