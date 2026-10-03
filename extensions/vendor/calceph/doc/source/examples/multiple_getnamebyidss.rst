.. ifconfig:: calcephapi in ('C')

    ::

         t_calcephcharvalue emb;
         t_calcephbin *peph;
 
         /* open the ephemeris file */
         peph = calceph_open("example1.dat");
         if (peph)
         {
            /* print the name of the Earth-Moon barycenter using the old numbering system */
            if (calceph_getnamebyidss(peph, 13, 0, &emb)) 
               printf("EMB=%d\n", emb);
            /* print the name of the Earth-Moon barycenter using the NAIF identification numbering system */
            if (calceph_getidbyname(peph, 3, CALCEPH_USE_NAIFID, &emb)) 
               printf("EMB=%d\n", emb);

            /* close the ephemeris file */
            calceph_close(peph);
         }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) emb
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) name_c
           TYPE(C_PTR) :: peph
           
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
               ! print the name of the Earth-Moon barycenter using the old numbering system
               ! remove the c null character
               if (calceph_getidbyname(peph, 13, 0, emb).eq.1) then
                   emb = name_c(1:index(fname, C_NULL_CHAR)-1)
                   write (*,*) "EMB=", emb
               endif
               ! print the name of the Earth-Moon barycenter using the NAIF identification numbering system
               ! remove the c null character
               if (calceph_getidbyname(peph, 3, CALCEPH_USE_NAIFID, emb).eq.1) then
                   emb = name_c(1:index(fname, C_NULL_CHAR)-1)
                   write (*,*) "EMB=", emb
               endif
               call calceph_close(peph)
            endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           character(len=CALCEPH_MAX_CONSTANTVALUE) emb
           
           res = f90calceph_open(peph, "example1.dat")
           if (res.eq.1) then
              !  print the name of the Earth-Moon barycenter using the old numbering system
              if (f90calceph_getidbyname(peph, 13, 0, emb).eq.1) then
                   write (*,*) "EMB=", emb
               endif
              ! print the name of the Earth-Moon barycenter using the NAIF identification numbering system
              if (f90calceph_getidbyname(peph,3, CALCEPH_USE_NAIFID, emb).eq.1) then
                   write (*,*) "EMB=", emb
              endif 
             call f90calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import *
        
        peph = CalcephBin.open("example1.dat")
        #  print the name of the Earth-Moon barycenter using the old numbering system
        emb = peph.getnamebyidss(13, 0)
        print(emb)
        # print the name of the Earth-Moon barycenter using the NAIF identification numbering system
        emb = peph.getnamebyidss(3, Constants.USE_NAIFID)
        print(emb)
        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open('example1.dat');
        # print the name of the Earth-Moon barycenter using the old numbering system
        emb = peph.getnamebyidss(13, 0)
        print(emb)
        # print the name of the Earth-Moon barycenter using the NAIF identification numbering system
        emb = peph.getnamebyidss(3, Constants.USE_NAIFID)
        print(emb)
        peph.close();

