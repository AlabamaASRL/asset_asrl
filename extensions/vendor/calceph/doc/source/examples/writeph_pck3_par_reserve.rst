The following example creates a new ephemeris file ephemeris.bsp with segment of type 3 
for the orientation of two bodiesusing OpenMP.

.. ifconfig:: calcephapi in ('C')

    ::

         t_writephbin *weph;

         weph = writeph_pck_create("ephemeris.bpc","planet_2", 0);
         if (weph)
         {
               int frame = 1; /* ICRF */
               int degree = 12;
               int target_count = 2;
               int targets[2] = { 3000099, 4000099 };
               int records_count[2] = { 10, 20 };
               int len_timespan[2] = { 32, 16 };
               const char segids[2] = { "seg_body1",  "seg_body2" };
               double jd_start = 2460000;
               double jd_end = 2460000+32*10;
               int reservation;

               reservation = writeph_pck3_seq_reserve(weph, target_count, targets, 
                                      frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, record_counts, degree+1, segids);

               #pragma omp parallel for
               for (int body = 0; body<target_count; body++)
               {
                    #pragma omp parallel for
                    for (int k=0; k<records_count[body]; k++)
                    {
                        double coefs[78];  /*  size = 13*6  = (degree+1) * 6 components */

                        double jd_startk = jd_start+k*len_timespan[body];
                        double jd_endk = jd_start+(k+1)*len_timespan[body];


                        /* ... fill the array coefs with the coefficients of the Chebychev polynomials ...
                            coefs[..] =...
                        */
                        writeph_pck3_par_write(weph, reservation, body,  k, 1,  coefs);
                    }
               }
                

               writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    ::

          USE, INTRINSIC :: ISO_C_BINDING
          use calceph
          TYPE(C_PTR) :: weph
          INTEGER len_timespan
          REAL(8) :: jd_start, jd_end
          INTEGER frame, degree, body, k, ret
          REAL(8), dimension(78) :: coefs !  size = 13*6  = (degree+1) * 6 components
          INTEGER, dimension(2) :: record_counts, targets
          INTEGER target_count, reservation
          REAL(8), dimension(2) :: len_timespan 
          CHARACTER(len=40), dimension (2) :: segids

          frame = 1 ! ICRF
          target_count = 2
          targets(1) = 3000099
          targets(2) = 4000099
          len_timespan(1) = 32 
          len_timespan(2) = 16 
          record_counts(1) = 10
          record_counts(2) = 20
          segids(1) = "seg_body1"//C_NULL_CHAR 
          segids(2) = "seg_body2"//C_NULL_CHAR 
          degree = 12
          jd_start = 2460000
          jd_end = jd_start+32*10

          weph = writeph_pck_create("ephemeris.bpc"//C_NULL_CHAR, "planet_2"//C_NULL_CHAR, 0)
          if (C_ASSOCIATED(weph)) then
          
               reservation = writeph_pck3_seq_reserve(weph, target_count, targets, 
                                      frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, record_counts, degree+1, segids)
               
               !$omp parallel for private(body)
               do body=0, target_count
                    !$omp parallel for private(k, coefs)
                    do k=0, record_counts(body)-1
                         ! ... fill the array coefs with the coefficients of the Chebychev polynomials ...
                         ! coefs(...) =...
                         ret = writeph_pck3_par_write(weph, reservation, body,  k, 1,  coefs)
                    enddo
               enddo
               call writeph_close(weph)    
          endif 
       

