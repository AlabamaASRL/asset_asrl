
.. ifconfig:: calcephapi in ('C')

    ::

        int res, timescale;
        calceph_sopen("example1.dat");
        timescale = calceph_sgettimescale();
        printf("timescale : %s\n", timescale==1?"TDB":"TCB");

