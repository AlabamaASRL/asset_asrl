.. ifconfig:: calcephapi in ('C')

    The following example adds a comment to the new ephemeris file ephemeris.bsp
    
    ::

         t_writephbin *weph;

         weph = writeph_spk_create("ephemeris.bsp","asteroid_1");
         if (weph)
         {
           writeph_comment(weph, "Created by ...\nDate : ....\n");
           
           /* ... compute and write to weph ... 
           */
           writeph_close(weph);
         }

.. ifconfig:: calcephapi in ('F2003')

    The following example adds a comment to the new ephemeris file ephemeris.bsp

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph

           weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "asteroid_1"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ret = writeph_comment(weph, "Created by ..."//C_NULL_CHAR)
                ret = writeph_comment(weph, "Date  ..."//C_NULL_CHAR)

                ! ... compute and write to weph ... 
           
                call writeph_close(weph)    
           endif 

