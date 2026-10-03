

::

    int res;
    char version[CALCEPH_MAX_CONSTANTVALUE];
    calceph_sopen("example1.dat");
    res = calceph_sgetfileversion(version);
    printf("The version of the file is %s\n", version);
