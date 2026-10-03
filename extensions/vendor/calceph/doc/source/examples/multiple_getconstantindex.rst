.. ifconfig:: calcephapi in ('C')

    ::

     int j;
     char nameconstant[CALCEPH_MAX_CONSTANTNAME];
     double valueconstant;
     t_calcephbin *peph;
 
     /* open the ephemeris file */
     peph = calceph_open("example1.dat");
     if (peph)
     {
        for (j=1; j<=calceph_getconstantcount(peph); j++)
        {
            calceph_getconstantindex(peph, j, nameconstant, &valueconstant);
            printf("'%s'\t= %23.16E\n", nameconstant, valueconstant);
        }

        /* close the ephemeris file */
        calceph_close(peph);
     }

.. ifconfig:: calcephapi in ('F2003')

    ::

           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           integer j
           real(8) valueconstant
           character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant
           TYPE(C_PTR) :: peph
           
        ! open the ephemeris file 
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then

        ! print the list of constants 
             do j=1, calceph_getconstantcount(peph)
               res = calceph_getconstantindex(peph,j,nameconstant, valueconstant)
               write (*,*) nameconstant,"=",valueconstant
             enddo

             call calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           integer n
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
             do j=1, f90calceph_getconstantcount(peph)
                res = f90calceph_getconstantindex(peph,j,nameconstant,valueconstant)
                write (*,*) nameconstant,"=",valueconstant
             enddo
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        n = peph.getconstantcount()
        for j in range(1, n+1):
            name, value = peph.getconstantindex(j)
            print(name, value)
        
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        n = peph.getconstantcount()
        for j=1:n
            [ name, value ] = peph.getconstantindex(j)
        end
        peph.close();
        
