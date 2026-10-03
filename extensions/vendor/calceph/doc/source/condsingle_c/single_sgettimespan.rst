

::

    int res;
    double jdfirst, jdlast;
    int cont;
    calceph_sopen("example1.dat");
    res = calceph_sgettimespan(&jdfirst, &jdlast, &cont);
    printf("data available between [ %f, %f ]. continuous=%d\n", 
           jdfirst, jdlast, cont);

