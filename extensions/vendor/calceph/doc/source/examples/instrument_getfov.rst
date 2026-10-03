.. ifconfig:: calcephapi in ('C')

    ::

        t_calcephbin *peph;
        int instrumentid = -42550;
        int shape;
        t_calcephcharvalue frame;
        double vector[3];
        double boundary[3];

        /* open the kernel file */
        peph = calceph_open("example_ik.ti");
    
        if (peph != NULL)
        {
            /* get the field of view */
            if (calceph_getfov(peph, instrumentid, &shape, &frame, vector, boundary, 3) != 0)
            {
                printf("Instrument %d\n", instrumentid);
                printf("Shape=%d Frame=%s\n", shape, frame);
                printf("Boresight vector : %f %f %f\n", vector[0], vector[1], vector[2]);
                printf("Boundary vector : %f %f %f\n", boundary[0], boundary[1], boundary[2]);
            }

            /* close the file */
            calceph_close(peph);
        }


.. ifconfig:: calcephapi in ('F2003')

    ::
    
        integer res, instrumentid
        real(8), dimension(1:3) :: boresight_vector
        real(8), dimension(1:3) :: boundary_vector
        character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) frame
        TYPE(C_PTR) :: peph


        peph = calceph_open("example_ik.ti"//C_NULL_CHAR)
        if (C_ASSOCIATED(peph)) then
            
            instrumentid = -42550
            res = calceph_getfov(peph, instrumentid, shape, frame, boresight_vector, boundary_vector, 3)

            write(*,*) "Instrument ", instrumentid
            write(*,*) "Shape", shape
            write(*,*) "Frame", frame
            write(*,*) "Boresight vector", boresight_vector
            write(*,*) "Boundary vector", boundary_vector

            call calceph_close(peph)
        endif


.. ifconfig:: calcephapi in ('F90')

    ::
    
        integer*8 peph
        character(len=CALCEPH_MAX_CONSTANTVALUE) UNIT
        integer res, instrumentid
        double precision boresight_vector(3)
        double precision boundary_vector(3)

        res = f90calceph_open(peph, "example_ik.ti")
        if (res.eq.1) then

            instrumentid = -42550
            res = f90calceph_getfov(peph, instrumentid, shape, frame, boresight_vector, boundary_vector, 3)

            write(*,*) "Instrument ", instrumentid
            write(*,*) "Shape", shape
            write(*,*) "Frame", frame
            write(*,*) "Boresight vector", boresight_vector
            write(*,*) "Boundary vector", boundary_vector

            call f90calceph_close(peph)
        endif



.. ifconfig:: calcephapi in ('Python')

    ::

        from calcephpy import *
        peph = CalcephBin.open("example_ik.ti")
        instrumentid = -42550
        shape, frame, boresight_vector, boundary_vectors = peph.getfov(instrumentid)
        print("shape=",shape) # print shape= 3
        print("frame=",frame) # print frame= EXAMPLE_CIRCLE
        print("boresight_vector=",boresight_vector) # print boresight_vector= [0.0, 0.0, 25.0]
        print("boundary_vectors=",boundary_vectors) # print boundary_vectors= [-3.046733585128687, 0.0, 24.81365379103305]
        peph.close()




.. ifconfig:: calcephapi in ('Mex')

    ::

        peph = CalcephBin.open("example_ik.ti")
        instrumentid = -42550
        [shape, frame, boresight_vector, boundary_vectors] = peph.getfov(instrumentid)
        # print shape = 3
        # print frame = EXAMPLE_CIRCLE
        # print boresight_vector = 0    0   25
        # print boundary_vectors = -3.0467         0   24.8137
        peph.close()                        
