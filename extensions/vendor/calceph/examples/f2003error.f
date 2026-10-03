!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003error.f 
!  \brief Example of usage of the error handler functions
!   with a fortran compiler compliant with the fortran 2003 standard
!         This example generates errors.
!         In a first step, on error, it displays a custom message using 
!          its own subroutine.
!         In a second step, on error, the library displays the default error message 
!          but continues the execution.
!         In a last step, on error, the library displays the default error message 
!          but exits.
!         So the message "goodbye" is never displayed.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, CNRS
!   email of the author : Mickael.Gastineau@obspm.fr
!
!*/
!/*-----------------------------------------------------------------*/

!/*-----------------------------------------------------------------*/
!/* License  of this file :
!  This file is "triple-licensed", you have to choose one  of the three licenses 
!  below to apply on this file.
!  
!     CeCILL-C
!     	The CeCILL-C license is close to the GNU LGPL.
!     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
!   
!  or CeCILL-B
!        The CeCILL-B license is close to the BSD.
!        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
!  
!  or CeCILL v2.1
!       The CeCILL license is compatible with the GNU GPL.
!       ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
!  
! 
! This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under 
! French law and abiding by the rules of distribution of free software.  
! You can  use, modify and/ or redistribute the software under the terms 
! of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA  
! at the following URL "http://www.cecill.info". 
!
! As a counterpart to the access to the source code and  rights to copy,
! modify and redistribute granted by the license, users are provided only
! with a limited warranty  and the software's author,  the holder of the
! economic rights,  and the successive licensors  have only  limited
! liability. 
!
! In this respect, the user's attention is drawn to the risks associated
! with loading,  using,  modifying and/or developing or reproducing the
! software by the user in light of its specific status of free software,
! that may mean  that it is complicated to manipulate,  and  that  also
! therefore means  that it is reserved for developers  and  experienced
! professionals having in-depth computer knowledge. Users are therefore
! encouraged to load and test the software's suitability as regards their
! requirements in conditions enabling the security of their systems and/or 
! data to be ensured and,  more generally, to use and operate it in the 
! same conditions as regards security. 
!
! The fact that you are presently reading this means that you have had
! knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
!*/
!/*-----------------------------------------------------------------*/

!/*-----------------------------------------------------------------*/
!/* custom error handler */
!/*-----------------------------------------------------------------*/
        subroutine myhandler(msg, msglen) BIND(C)
           USE, INTRINSIC :: ISO_C_BINDING
           implicit none
           character(kind=C_CHAR), dimension(msglen), intent(in)        &
     &                  :: msg
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
                character(kind=C_CHAR), dimension(msglen), intent(in)   &
     &                  :: msg
                integer(C_INT), VALUE, intent(in) :: msglen
            end subroutine
           end interface 

! this date isnot in the example ephemeris file
           jd0 = -2442457D10
           dt = 0.5E0
! open the ephemeris file 
           res = calceph_sopen("example1.dat"//C_NULL_CHAR)
           if (res.eq.1) then
            write (*,*) "Step 1 :"
            write (*,*) "it displays a custom message "
            write (*,*) "using its own subroutine"
            write (*,*) "This subroutine has an explicit interface"
            write (*,*) "(see at line 98-101 in the f2003error.f)"
! set the  error handler to use my own callback 
            call calceph_seterrorhandler(3, c_funloc(myhandler))
! call with an invalid time
            res = calceph_scompute(jd0, dt, 10, 3, PV)
            
            write (*,*) 
            write (*,*) 
            write (*,*) "Step 2 :"
            write (*,*) "calceph library displays the message "
            write (*,*) "but it continues"
! set the error handler to display error and continue
            call calceph_seterrorhandler(1, C_NULL_FUNPTR)
! call with an invalid time
            res = calceph_scompute(jd0, dt, 10, 3, PV)
            
            write (*,*) 
            write (*,*) 
            write (*,*) "Step 3 :"
            write (*,*) "calceph library displays the message "
            write (*,*) "but the programs exits"
! set the  error handler to stop on error
            call calceph_seterrorhandler(2, C_NULL_FUNPTR)
! call with an invalid time
            res = calceph_scompute(jd0, dt, 10, 3, PV)

            write (*,*) "Goodbye"
            write (*,*) "If you see this message, you find a bug !"
! close the ephemeris file 
           call calceph_sclose
        else
           write (*,*) "The ephemeris can't be opened"
        endif
       stop
       end
      