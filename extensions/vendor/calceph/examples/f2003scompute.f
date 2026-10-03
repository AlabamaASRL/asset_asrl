!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003scompute.f 
!  \brief simple computation with a file and a date provided on the command line
!   e.g.: cscompute  myfile.dat 2.451545E+06
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, CNRS
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
!/* print coordinates */
!/*-----------------------------------------------------------------*/
       subroutine printcoord(PV,name)
           implicit none
           real(8), intent(in):: PV(6)
           character(len=*), intent(in) :: name
           integer j
    
           write(*,*) name, " :"
           do j=1,6
               write(*,*) PV(j)
           enddo
           write(*,*) 
       end subroutine

!/*-----------------------------------------------------------------*/
!/* main program */
!/*-----------------------------------------------------------------*/
       program f2003scompute
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           real(8) jd0
           real(8) dt
           real(8) PV(6)
           integer cmd_count
           character(len=2048) filename, jd0str
           integer status, length
           
! get the command line 
           cmd_count = COMMAND_ARGUMENT_COUNT()
           if (cmd_count.ne.2) then
              write(*,*) "Usage : cscompute filename juliandate"
              write(*,*) 
              write(*,*) "e.g.  : cscompute inpop08a.dat 2.451545E+06"
              stop
           endif
           call GET_COMMAND_ARGUMENT(1, filename, length, status)
           call GET_COMMAND_ARGUMENT(2, jd0str, length, status)
            
           read(jd0str,*) jd0
           write(*,*) "julian date : ", jd0
           
           dt = 0.0E0
! open the ephemeris file 
           res = calceph_sopen(trim(filename)//C_NULL_CHAR)
           if (res.eq.1) then

! the heliocentric coordinates of the Earth 
          res = calceph_scompute(jd0, dt, 3, 11, PV)
          if (res.eq.1) then
           call printcoord(PV,"heliocentric coordinates of the Earth")
          endif
! close the ephemeris file 
          call calceph_sclose
        else
           write (*,*) "The ephemeris file can't be opened"
        endif
       stop
       end
      
