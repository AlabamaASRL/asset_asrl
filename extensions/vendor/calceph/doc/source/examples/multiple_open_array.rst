.. ifconfig:: calcephapi in ('C')

    ::
    
        const char *filear[2]= {"example1.bsp", "example1.tpc"};
        t_calcephbin *peph;
        
        peph = calceph_open_array(2,filear);
        if (peph)
        {
            /* ... computation ...*/
            calceph_close(peph);
        }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
        TYPE(C_PTR) :: peph
        character(len=256), dimension (2) :: filear
        filear(1) = "example1.bsp"//C_NULL_CHAR
        filear(2) = "example1.tpc"//C_NULL_CHAR
        peph = calceph_open_array(2, filear, 256) 
        if (C_ASSOCIATED(peph)) then
          res = calceph_prefetch(peph)
          ! ...  computation ...
          call calceph_close(peph)
        endif

.. ifconfig:: calcephapi in ('F90')

    ::
    
           integer*8 peph
           integer res
           character*80 filear(2)

           filear(1) = "example1.bsp"
           filear(2) = "example1.tpc"
           res = f90calceph_open_array(peph, 2, filear, 80)
           if (res.eq.1) then
              res = f90calceph_prefetch(peph)
              !  ...  computation ...
              call calceph_close(peph)
           endif


.. ifconfig:: calcephapi in ('Python')

    ::
    
        from calcephpy import CalcephBin
        peph = CalcephBin.open(['example1.bsp', 'example1.tpc'])

        # ... computation ...

        peph.close()


.. ifconfig:: calcephapi in ('Mex')

    ::
    
        peph = CalcephBin.open(cellstr({'example1.bsp', 'example1.tpc'}))

        % ... computation ...

        peph.close()

