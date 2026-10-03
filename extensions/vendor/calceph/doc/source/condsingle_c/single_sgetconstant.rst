
::

    int res;
    double UA;
    calceph_sopen("example1.dat");
    res = calceph_sgetconstant("UA",&UA);
    if (res) 
    {
       printf("astronomical unit=%23.16E\n", UA);
    }
