The following example writes a first part to the ephemeris file ephemeris.bsp and continue from a restart checkpoint.

.. ifconfig:: calcephapi in ('C')

    ::

         t_writephbin *weph;

         weph = writeph_spk_create("ephemeris.bsp","asteroid_1");
         if (weph)
         {
           /* 
             ...  computation and writing to weph ... 
           */
           writeph_dump(weph, "checkpoint_ephemeris_bsp.dump")
            /* 
              ... compute and write to weph ...
              ... may be an interruption here ...
           */
           writeph_close(weph);
         }
         
         /* may be an interruption here */

         weph = writeph_spk_open("ephemeris.bsp");
         if (weph)
         {
           /* restart from the checkpoint */
           writeph_restore(weph, "checkpoint_ephemeris_bsp.dump");

           /* 
            ... compute and write to weph ...
           */
           writeph_close(weph);
         }
         


.. ifconfig:: calcephapi in ('F2003')


    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           TYPE(C_PTR) :: weph
           INTEGER(C_INT) :: ret
           
           weph = writeph_spk_create("ephemeris.bsp"//C_NULL_CHAR, "asteroid_1"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ! ... compute and write to weph ...

                ret = writeph_dump(weph, "checkpoint_ephemeris_bsp.dump")
           
                ! ... compute and write to weph ...
                ! ... may be an interruption here ....

                call writeph_close(weph)    
           endif 

           weph = writeph_spk_open("ephemeris.bsp"//C_NULL_CHAR)
           if (C_ASSOCIATED(weph)) then
           
                ! restart from the checkpoint 
                ret = writeph_restore(weph, "checkpoint_ephemeris_bsp.dump")
           
                ! ... compute and write to weph ...

                call writeph_close(weph)    
           endif 
