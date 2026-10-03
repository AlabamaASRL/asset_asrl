.. ifconfig:: calcephapi in ('C')

    The following example appends data to the ephemeris file ephemeris.bpc
    
    ::

         t_writephbin *weph;

         weph = writeph_pck_open("ephemeris.bpc");
         if (weph)
         {
           /* 
             ...  computation and writing to weph ... 
           */
           writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    The following example appends data the ephemeris file ephemeris.bpc

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph
           
           weph = writeph_pck_open("ephemeris.bpc"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ! ... computation ... 
           
                call writeph_close(weph)    
           endif 

