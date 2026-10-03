.. ifconfig:: calcephapi in ('C')

    ::

         char version[CALCEPH_MAX_CONSTANTVALUE];
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
           if (calceph_getfileversion(peph, version)) 
             printf("The version of the file is %s\n", version);

           /* close the ephemeris file */
           calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE) version
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
                res = calceph_getfileversion(peph, version)
                write (*,*) "The version of the file is ", version

               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE) version
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then

             res = f90calceph_getfileversion(peph, version)
             write (*,*) "The version of the file is ", version
 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        version = peph.getfileversion()
        print(version)
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        version = peph.getfileversion()
        peph.close();

