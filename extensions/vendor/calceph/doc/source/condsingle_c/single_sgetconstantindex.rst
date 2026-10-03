
::

     int j;
     int res;
     char nameconstant[CALCEPH_MAX_CONSTANTNAME];
     double valueconstant;

     res = calceph_sopen("example1.dat");
     if (res)
     {
         for (j=1; j<=calceph_sgetconstantcount(); j++)
         {
             calceph_sgetconstantindex(j, nameconstant, &valueconstant);
             printf("'%s'\t= %23.16E\n", nameconstant, valueconstant);
         }
         calceph_sclose();
     }
