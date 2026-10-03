!/*-----------------------------------------------------------------*/
!/*! 
!  \file f77single.f 
!  \brief Example of usage of the single file access functions
!   with a fortran compiler compliant with the fortran 77 standard
!         This example reads the constant EMRAT, AU, GM_Mer and print their values.
!         It computes for a date
!          the geocentric moon coordinates, 
!          the value TT-TDB 
!          the heliocentric coordinates of Mars.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, CNRS
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
           double precision PV(6)
           character(len=*)  name
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
       program f77single
           implicit none
           include 'f90calceph.h'
           integer res
           double precision AU, EMRAT, GM_Mer
           double precision jd0
           double precision dt
           double precision PV(6)
           integer j, t
           double precision valueconstant
           character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant
           integer cont
           double precision  jdfirst, jdlast

           jd0 = 2442457
           dt = 0.5E0
! open the ephemeris file 
           res = f90calceph_sopen("example1.dat")
           if (res.eq.1) then
               write (*,*) "The ephemeris is already opened"

! print the time span
               t = f90calceph_sgettimescale()
               if (t.eq.1) then   
                 write (*,*) "timescale : TDB"
               endif  
               if (t.eq.2) then   
                 write (*,*) "timescale : TCB"
               endif  
           if(f90calceph_sgettimespan(jdfirst, jdlast, cont).eq.1)then
             write(*,*) "data available between ",jdfirst,"and",jdlast
             write(*,*) "continuous data : ", cont
           endif

! print the values of AU, EMRAT and GM_Mer 
               if (f90calceph_sgetconstant("AU", AU).eq.1) then
                   write (*,*) "AU=", AU
               endif
               if (f90calceph_sgetconstant("EMRAT", EMRAT).eq.1) then
                   write (*,*) "EMRAT=", EMRAT
               endif
               if (f90calceph_sgetconstant("GM_Mer", GM_Mer).eq.1)then
                   write (*,*) "GM_Mer=", GM_Mer
               endif
 
! compute and print the coordinates 
! the geocentric moon coordinates 
              res = f90calceph_scompute(jd0, dt, 10, 3, PV)
              call printcoord(PV,"geocentric coordinates of the Moon")
! the value TT-TDB 
              if (f90calceph_scompute(jd0, dt, 16, 0, PV).eq.1) then
               write (*,*) "TT-TDB = ", PV(1)
              endif
! the heliocentric coordinates of Mars 
              res = f90calceph_scompute(jd0, dt, 4, 11, PV)
              call printcoord(PV,"heliocentric coordinates of Mars")
 
! print the whole list of the constants 
             write (*,*) "list of constants"
             do j=1, f90calceph_sgetconstantcount()
                res = f90calceph_sgetconstantindex(j,nameconstant,        &
     &                                       valueconstant)
                write (*,*) nameconstant,"=",valueconstant
             enddo

! close the ephemeris file 
               call f90calceph_sclose
               write (*,*) "The ephemeris is already closed"
            else
               write (*,*) "The ephemeris can't be opened"
            endif
       stop
       end
      
