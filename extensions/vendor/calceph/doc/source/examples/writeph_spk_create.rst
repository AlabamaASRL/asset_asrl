.. ifconfig:: calcephapi in ('C')

    The following example creates a new ephemeris file ephemeris.bsp
    
    ::

         t_writephbin *weph;

         weph = writeph_spk_create("ephemeris.bsp","asteroid_1");
         if (weph)
         {
           
           /* ... compute and write to weph ... */
           writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    The following example creates a new ephemeris file ephemeris.bsp

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph
           
           weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "asteroid_1"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ! ... compute and write to weph ... 
           
                call writeph_close(weph)    
           endif 

