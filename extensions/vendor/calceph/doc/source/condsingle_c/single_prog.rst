
::

    #include <stdio.h>
    #include "calceph.h"

    /*-----------------------------------------------------------------*/
    /* main program */
    /*-----------------------------------------------------------------*/
    int main()
    {
     int res;
     double AU, EMRAT, GM_Mer;
     double jd0=2451624;
     double dt=0.5E0;
     double PV[6];

     /* open the ephemeris file */
     res = calceph_sopen("example1.dat");
     if (res)
     {
        printf("The ephemeris is already opened\n");
       /* print the values of AU, EMRAT and GM_Mer */
       if (calceph_sgetconstant("AU", &AU)) 
           printf("AU=%23.16E\n", AU);

       if (calceph_sgetconstant("EMRAT", &EMRAT)) 
         printf("EMRAT=%23.16E\n", EMRAT);

       if (calceph_sgetconstant("GM_Mer", &GM_Mer)) 
         printf("GM_Mer=%23.16E\n", GM_Mer);

       /* compute and print the coordinates */
       /* the geocentric moon coordinates in AU and AU/day */
       calceph_scompute(jd0, dt, 10, 3, PV);
       printcoord(PV,"geocentric coordinates of the Moon in AU and AU/day");

       /* the value TT-TDB */
       calceph_scompute(jd0, dt, 16, 0, PV);
       printf("TT-TDB = %23.16E\n", PV[0]);

       /* the heliocentric coordinates of Mars */
       calceph_scompute(jd0, dt, 4, 11, PV);
       printcoord(PV,"heliocentric coordinates of Mars");

       /* close the ephemeris file */
       calceph_sclose();
       printf("The ephemeris is already closed\n");
     }
     else
     {
       printf("The ephemeris can't be opened\n");
     }
     return res;
    }
