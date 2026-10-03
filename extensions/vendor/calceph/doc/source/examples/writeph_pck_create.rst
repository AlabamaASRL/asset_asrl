.. ifconfig:: calcephapi in ('C')

    The following example creates and writes the ephemeris file ephemeris.bpc
    
    ::

         t_writephbin *weph;

         weph = writeph_pck_create("ephemeris.bpc","asteroid_1", 0);
         if (weph)
         {
           /* 
             ...  computation and writing to weph ... 
           */
           writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    The following example creates and writes the ephemeris file ephemeris.bpc

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph
           
           weph = writeph_pck_create("ephemeris.bpc"//C_NULL_CHAR, "asteroid_1"//C_NULL_CHAR, 0)
           if (C_ASSOCIATED(weph)) then
           
                ! ... computation ... 
           
                call writeph_close(weph)    
           endif 

