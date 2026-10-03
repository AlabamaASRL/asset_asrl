The following example creates a new ephemeris file ephemeris.bsp with segment of type 3 
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
               int degree = 12;
               double jd_start = 2460000;
               double jd_end = 2460300;
               double coefs[80];  /*  size = 13*6+2  =  (degree+1) * 6+2 components */
               double epoch[1];   /*  size = 1  date only */

               writeph_spk14_begin(weph, NAIFID_VENUS, NAIFID_SUN, frame, jd_start, 0.0, jd_end, 0.0, 
                                   degree+1, segid);

               for (int k=0; k<100; k++)
               {
                    
                    /* ... fill the array coefs with the epochs and coefficients of the Chebychev polynomials ...
                         epoch[k] = ...
                         coefs[..] =...
                    */
                    /* write one record  only */
                    writeph_spk14_add(weph, 1, coefs, epochs);
               }
               writeph_spk14_end(weph);
               

               writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    ::

          USE, INTRINSIC :: ISO_C_BINDING
          use calceph
          TYPE(C_PTR) :: weph
          INTEGER len_timespan
          REAL(8) :: jd_start, jd_end
          INTEGER frame, degree, k, ret
          REAL(8), dimension(78) :: coefs !  size = 78 =  (degree+1) * 6 components
          REAL(8), dimension(1) :: epochs !  size = 1 =  one date only
          
          len_timespan = 32 ! days 
          frame = 1 ! ICRF
          degree = 12
          jd_start = 2460000
          jd_end = 2460300

          weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "planet_2"//C_NULL_CHAR, 0)
          if (C_ASSOCIATED(weph)) then
          
                ret = writeph_spk14_seq_write(weph, NAIFID_VENUS, NAIFID_SUN, frame, jd_start, 0.0, jd_end, 0.0, 
                              degree+1, "seg_planet"//C_NULL_CHAR)

                do k=0, 100
                    ! ... fill the array coefs with the coefficients of the Chebychev polynomials ...
                    ! coefs(...) =...
                    ! epochs(...) = ...
                    ret = writeph_spk14_add(weph, 1, coefs, epochs)
               enddo
               ret = writeph_spk14_end(weph)
               
               call writeph_close(weph)    
          endif 

