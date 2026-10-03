The following example creates a new ephemeris file ephemeris.bsp with segment of type 2 
for the orientation of one body, in the timescale TCB.

.. ifconfig:: calcephapi in ('C')

    ::

         t_writephbin *weph;
         const char segid[] = "seg_planet";

         weph = writeph_pck_create("ephemeris.bpc","planet_2", 0);
         if (weph)
         {
               int len_timespan = 32; /* days */
               int frame = 1; /* ICRF */
               int record_count = 10;
               int degree = 12;
               double jd_start = 2460000;
               double jd_end = jd_start+record_count*len_timespan;
               double coefs[390];  /*  size = 10*13*3  = record_count * (degree+1) * 3 components */

               for (int k=0; k<record_count; k++)
               {

                    double jd_startk = jd_start+k*len_timespan;
                    double jd_endk = jd_start+(k+1)*len_timespan;


                    /* ... fill the array coefs with the coefficients of the Chebychev polynomials ...
                         coefs[..] =...
                    */
               }
               writeph_pck102_seq_write(weph, 4000099, frame, jd_start, 0.0, jd_end, 0.0, 
                                      len_timespan, coefs, record_count, degree+1, segid);
               

               writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    ::

          USE, INTRINSIC :: ISO_C_BINDING
          use calceph
          TYPE(C_PTR) :: weph
          INTEGER len_timespan
          REAL(8) :: jd_start, jd_end
          INTEGER frame, record_count, degree, k, ret
          REAL(8), dimension(390) :: coefs !  size = 10*13*3  = record_count * (degree+1) * 3 components
          
          len_timespan = 32 ! days 
          frame = 1 ! ICRF
          record_count = 10
          degree = 12
          jd_start = 2460000
          jd_end = jd_start+record_count*len_timespan

          weph = writeph_pck_create("ephemeris.bpc"//C_NULL_CHAR, "planet_2"//C_NULL_CHAR, 0)
          if (C_ASSOCIATED(peph)) then
          
               do k=0, record_count-1
                    ! ... fill the array coefs with the coefficients of the Chebychev polynomials ...
                    ! coefs(...) =...
               enddo
               ret = writeph_pck102_seq_write(weph, 4000099, frame, jd_start, 0.0, jd_end, 0.0, 
                              len_timespan, coefs, record_count, degree+1, "seg_planet"//C_NULL_CHAR)
               
               call writeph_close(peph)    
          endif 

