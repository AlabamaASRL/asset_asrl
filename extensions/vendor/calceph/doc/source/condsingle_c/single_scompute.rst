
::
    
     int res;
     int j;
     double jd0=2451624;
     double dt1=0.5E0;
     double dt2=0.9E0;
 
     double PV[6];
     /* open the ephemeris file */
     res = calceph_sopen("example1.dat");
     if (res)
     {
       /* the heliocentric coordinates of Mars */
       calceph_scompute(jd0, dt1, 4, 11, PV);
       for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);
   
       calceph_scompute(jd0, dt2, 4, 11, PV);
       for(j=0; j<6; j++) printf("%23.16E\n", PV[j]);
 
       /* close the ephemeris file */
       calceph_sclose();
     }

