!/*-----------------------------------------------------------------*/
!/*! 
!  \file f77mcheck.f 
!  \brief Check the output of the function calceph_compute with a reference file.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,CNRS
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
!   or CeCILL-B
!       The CeCILL-B license is close to the BSD.
!       (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
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
!/* main function : prefetch=1 : prefetch the data to memory */
!/*-----------------------------------------------------------------*/
       subroutine fmaincheck(prefetch)
           implicit none
           include 'f90calceph.h'
           integer, intent(in) :: prefetch
           integer res, res2
           integer n, target1, center, status, j
           integer fileref 
           DATA fileref /12/
           double precision seuil
           double precision jd0
           double precision  dt
           double precision PV(6), PVcheck(6)
           integer*8 :: peph
           
           include 'fopenfiles.h'
           res = 0
           
! open refernce data file
           open(fileref, file=trim(TOPSRCDIR)                           &
     &      //"example1_tests.dat",status="old")
           
! open the ephemeris file 
           res2=f90calceph_open(peph,trim(TOPSRCDIR)                    &
     &      //"../examples/example1.dat")
           if (res2.eq.1) then
             if (prefetch.eq.1) then
              res2 = f90calceph_prefetch(peph)
             endif
             
             status = 0
             do while((status.eq.0).and.(res.eq.0))
! read the reference filejd0, target1, center, 
              read(fileref, *,iostat=status) jd0, target1, center,        &
     &              (PVcheck(j),j=1,6)
              if (status.eq.0) then
              
               dt = jd0-int(jd0)
               jd0 = int(jd0)+2.4515450000000000000D+06
! compute  the coordinates 
               res2=f90calceph_compute(peph,jd0, dt,target1,center,PV)
               if (res2.eq.0) then
                res = 2
               endif
! check the coordinates               
               seuil=1D-8
               n=6
               if (target1.eq.16) then
                n=1
                seuil=1D-8
               endif
               if (target1.eq.15) then
                n=1
                seuil=1D-7
               endif
               do j=1,n
                if (abs(PV(j)-PVcheck(j))>=seuil) then
                 write(*,*) "failure for ",target1, center
                 write(*,*) "at time ", jd0, dt
                 write(*,*) "diff=",PVcheck(j)-PV(j),PVcheck(j),PV(j)
                 write(*,*) "prefetch =  ",prefetch
                 res = 1               
                endif
               enddo
              endif
              
             enddo
! close the ephemeris file 
             call f90calceph_close(peph)
            else
               write (*,*) "The ephemeris can't be opened"
               res = 2
            endif
       close(fileref)
       if (res.ne.0) then
        stop 2
       endif
       end
      
!/*-----------------------------------------------------------------*/
!/* main program */
!/*-----------------------------------------------------------------*/
       program f77mcheck
           implicit none
           call fmaincheck(0)
           call fmaincheck(1)
           stop
       end
