.. ifconfig:: calcephapi in ('C')

    The following example appends data to the ephemeris file ephemeris.bsp
    
    ::

         t_writephbin *weph;
         const char segid[] = "seg_asteroid_1";

         weph = writeph_spk_open("ephemeris.bsp");
         if (weph)
         {
           /* 
             ...  computation and writing to weph ... 
           */
           writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    The following example appends data the ephemeris file ephemeris.bsp

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph
           
           weph = writeph_spk_open("ephemeris.bsp"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ! ... computation ... 
           
                call writeph_close(weph)    
           endif 

