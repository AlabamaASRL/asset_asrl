.. ifconfig:: calcephapi in ('C')

    ::
    
        #include <stdio.h>
        #include "calceph.h"

        int main(void)
        {
         double AU;
         t_calcephbin *peph;
 
         peph = calceph_open("example1.dat");
         if (peph)
         {
           if (calceph_getconstant(peph, "AU", &AU)) 
           {
                printf("AU=%23.16E\n", AU);
           }
       
           calceph_close(peph);
         }
         return 0;
        }

.. ifconfig:: calcephapi in ('F2003')

    ::
    
       program f2003multiple
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           integer res
           real(8) AU
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
           
              if (calceph_getconstant(peph, "AU"//C_NULL_CHAR, AU).eq.1) then
                   write (*,*) "AU=", AU
              endif

              call calceph_close(peph)
            endif
       stop
       end
 
.. ifconfig:: calcephapi in ('F90')

    ::
    
       program F90example
           implicit none
           include 'f90calceph.h'
           integer res
           double precision AU
           integer*8 :: peph
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
             write (*,*) "The ephemeris is already opened"
             if (f90calceph_getconstant(peph, "AU", AU).eq.1) then
                   write (*,*) "AU=", AU
             endif
             call f90calceph_close(peph)
             write (*,*) "The ephemeris is already closed"
           endif
       stop
       end

.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example1.dat")
        AU = peph.getconstant("AU")
        jd0 = 2451542
        dt = 0.5
        PV = peph.compute_unit(jd0, dt, NaifId.MOON, NaifId.EARTH,
                               Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
        print(PV)
        peph.close()

.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open('example1.dat');
        AU = peph.getconstant('AU')
        jd0 = 2451542
        dt = 0.5
        PV = peph.compute_unit(jd0, dt, NaifId.MOON, NaifId.EARTH,
                               Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID);
        disp(PV)
        peph.close();
