.. ifconfig:: calcephapi in ('C')

    The following example shows how to stop the execution on the error.

    ::

        t_calcephbin * eph;
        
        /* set the  error handler to stop on error */
        calceph_seterrorhandler(2, 0);
        
        /* open the ephemeris file */
        eph = calceph_open("example1.dat");
        
        /*... computation ... */
    
    The following example shows how to define a custom error handler function.

    ::
    
        /*-----------------------------------------------------------------*/
        /* custom error handler */
        /*-----------------------------------------------------------------*/
        static void myhandler(const char *msg)
        {
                puts("The calceph calls the function myhandler");
                printf("The message contains %d characters\n", (int)strlen(msg));
                puts("The error message is :");
                puts("----------------------");
                puts(msg);
                puts("----------------------");
                puts("The error handler returns");
        }

        int main(void)
        {
            t_calcephbin * eph;
        
            /* set the  error handler to stop on error */
            calceph_seterrorhandler(3, myhandler);
        
            /* open the ephemeris file */
            eph = calceph_open("example1.dat");
        
            /*... computation ... */
            
            return 0;
        }    


.. ifconfig:: calcephapi in ('F2003')

    The following example shows how to stop the execution on the error.

    ::
    
       program f2003error
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           real(8) jd0
           real(8) dt
           real(8) PV(6)
                      
        ! set the  error handler to stop on error
            call calceph_seterrorhandler(2, C_NULL_FUNPTR)

        ! open the ephemeris file 
           res = calceph_sopen("example1.dat"//C_NULL_CHAR)
           ...
           
       stop
       end      

    The following example shows how to define a custom error handler function.

    ::
    
        !/*-----------------------------------------------------------------*/
        !/* custom error handler */
        !/*-----------------------------------------------------------------*/
             subroutine myhandler(msg, msglen) BIND(C)
                USE, INTRINSIC :: ISO_C_BINDING
                implicit none
                character(kind=C_CHAR), dimension(msglen), intent(in) :: msg
                integer(C_INT), VALUE, intent(in) :: msglen
                write (*,*) "The calceph calls the function myhandler"
                write (*,*) "The message contains ",msglen," characters"
                write(*,*) "The error message is :"
                write(*,*) "----------------------"
                write(*,*) msg
                write(*,*) "----------------------"
                write(*,*) "The error handler returns"
             end

        !/*-----------------------------------------------------------------*/
        !/* main program */
        !/*-----------------------------------------------------------------*/
             program f2003error
                 USE, INTRINSIC :: ISO_C_BINDING
                 use calceph
                 implicit none
                 integer res
                 real(8) jd0
                 real(8) dt
                 real(8) PV(6)
         
                 interface
                  subroutine myhandler(msg, msglen) BIND(C)
                      USE, INTRINSIC :: ISO_C_BINDING
                      implicit none
                      character(kind=C_CHAR), dimension(msglen), intent(in) &
            &          :: msg
                      integer(C_INT), VALUE, intent(in) :: msglen
                  end subroutine
                 end interface 

         
        ! set the  error handler to use my own callback 
                 call calceph_seterrorhandler(3, c_funloc(myhandler))

        ! open the ephemeris file 
                res = calceph_sopen("example1.dat"//C_NULL_CHAR)
           
           ! ...
           
             stop
             end

.. ifconfig:: calcephapi in ('F90')

    The following example shows how to stop the execution on the error.

    ::
    
       program f77error
           implicit none
           include 'f90calceph.h'
           integer*8 peph
           integer res
                      
           ! set the  error handler to stop on error
            call f90calceph_seterrorhandler(2, 0)

           ! open the ephemeris file 
           res = f90calceph_open(peph, "example1.dat")
           ! ...
       stop
       end      

    The following example shows how to define a custom error handler function.

    ::
    
        !-----------------------------------------------------------------
        ! custom error handler
        !-----------------------------------------------------------------
                subroutine myhandler(msg) 
                   implicit none
                   character(len=*) :: msg
                   write (*,*) "The calceph calls the function myhandler"
                   write (*,*) "The message contains ",len(msg)," characters"
                   write(*,*) "The error message is :"
                   write(*,*) "----------------------"
                   write(*,*) msg
                   write(*,*) "----------------------"
                   write(*,*) "The error handler returns"
                end

        !-----------------------------------------------------------------
        ! main program 
        !-----------------------------------------------------------------
        program f77error
           implicit none
           include 'f90calceph.h'
           integer res
           integer*8 peph
           external myhandler
           

            ! set the  error handler to use my own callback 
            call f90calceph_seterrorhandler(3, myhandler)
            
           ! open the ephemeris file 
           res = f90calceph_open(peph, "example1.dat")
           ! ...
       stop
       end      
           

.. ifconfig:: calcephapi in ('Python')

    The following example shows how to stop the execution on the error.

    ::
    
        from calcephpy import *

        #set the  error handler to stop on error 
        seterrorhandler(2, 0);

        # open the ephemeris file 
        peph = CalcephBin.open("example1.dat")
    
     
    The following example shows how to define a custom error handler function.

    ::

        from calcephpy import *

        #-----------------------------------------------------------------
        # custom error handler
        #-----------------------------------------------------------------
        def myhandler(msg):
            print("The calceph calls the function myhandler");
            print("The message contains {0} characters\n".format(len(msg)))
            print("The error message is :")
            print("----------------------")
            print(msg)
            print("----------------------")
            print("The error handler returns")

        # set the  error handler to use my own callback 
        seterrorhandler(3, myhandler)
    
        # open the ephemeris file 
        peph = CalcephBin.open("example1.dat")



.. ifconfig:: calcephapi in ('Mex')

    The following example shows how to define a custom error handler function.

    ::

        %-----------------------------------------------------------------
        % custom error handler
        %-----------------------------------------------------------------
        function myhandler(msg)
            disp('The calceph calls the function myhandler');
            disp('The error message is :')
            disp('----------------------')
            disp(msg)
            disp('----------------------')
            disp('The error handler returns')
        end
         
        % set the  error handler to use my own callback 
        calceph_seterrorhandler(3, 'myhandler')
    
        % open the ephemeris file 
        peph = CalcephBin.open('example1.dat')

