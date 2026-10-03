!/*-----------------------------------------------------------------*/
!/*! 
!  \file f77mgetfov.f 
!  \brief Check if calceph_getfov_... works with fortran 77 compiler.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2025-2026, CNRS
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
!       The CeCILL-B license is close to the BSD.
!       ( http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
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
!/* main program */
!/*-----------------------------------------------------------------*/
       program f77getfov
           implicit none
           include 'f90calceph.h'
           integer*8 peph
           integer r1, j, res
           integer nbounds, shape
           character(len=CALCEPH_MAX_CONSTANTVALUE) :: frame
           real*8 vector(3)
           real*8 arraybounds(1:256)
           real*8, parameter :: expected_rectangle(1:12)                &
     &      = (/0.063657D0,0.000004D0,0.997972D0,                       &
     &      0.063657D0, -0.000004D0, 0.997972D0,                        &
     &      0.063666D0, -0.000004D0, 0.997971D0,                        &
     &      0.063666D0, 0.000004D0, 0.997971D0/)
           real*8, parameter :: expected_vector(1:3)                    &
     &      = (/0.0636614381316129D0, 0D0, 0.997971553349531D0/)
           include 'fopenfiles.h'
           
           res=f90calceph_open(peph, trim(TOPSRCDIR)//"example_ik.ti")
           if (res.eq.1) then
                    
        nbounds=f90calceph_getfov(peph,-42552,shape,frame,vector,        &
     &    arraybounds, 0)
          write(*,*) "nbounds=", nbounds
       
        r1 = f90calceph_getfov(peph,-42552,shape,frame,vector,           &
     &    arraybounds, nbounds)
        if ((r1.ne.4) .or. (shape.ne.2).or. (nbounds.ne.4) .or.          &
     &   (frame.ne."EXAMPLE_RECTANGLE")) then
          write(*,*) "nbounds=", nbounds
          write(*,*) "r1=", r1
          write(*,*) "shape=", shape
          write(*,*) "frame=", frame
          write(*,*) "vector=", vector
          write(*,*) "arraybounds=", arraybounds(1:3*nbounds)
          r1=0
        endif
        do j=1,12
         if(abs(arraybounds(j)-expected_rectangle(j))>1D-6) then
          write(*,*) "nbounds=", nbounds
          write(*,*) "arraybounds=", arraybounds(1:3*nbounds)
          r1=0
         endif
        enddo 
       
        do j=1,3
         if(abs(vector(j)-expected_vector(j))>1D-6) then
          write(*,*) "vector=", vector(1:3)
          r1=0
         endif
        enddo 
               
        call f90calceph_close(peph)       
        if(r1.eq.4)then
            stop
        endif
           endif
       write(*,*) 'stop with a failure'
       stop 2    
       end
      