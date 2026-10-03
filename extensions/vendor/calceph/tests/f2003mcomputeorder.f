!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003mcomputeorder.f 
!  \brief Check the output of the function calceph_compute_order
!  and calceph_orient_order for the order 0 and 1
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2016, 2017, CNRS
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
! check the array PV and PVcheck
!/*-----------------------------------------------------------------*/
      subroutine  checkoutput(n, PV, PVcheck,seuil,jd0, dt,center,        &
     &                       vtarget, order, res, unit)
           implicit none
           integer, intent(in) :: n, center,vtarget, order
           integer, intent(inout) :: res
           real(8), intent(in) :: PV(12), PVcheck(12)
           real(8), intent(in) :: seuil, jd0, dt
           character(len=*), intent(in) :: unit
           integer j

               do j=1, n
                 if (abs(PV(j)-PVcheck(j))>=seuil) then
                  write(*,*) "failure for ",vtarget, center
                  write(*,*) "at time ", jd0, dt
                  write(*,*) "unit=",unit
                  write(*,*) "order=",order
                  write(*,*) "diff=",PVcheck(j)-PV(j),PVcheck(j),PV(j)
                  write(*,*) PVcheck
                  write(*,*) PV
                  res = 2               
                 endif
                enddo
             
           return
           end 

!/*-----------------------------------------------------------------*/
! check the array PV and PVcheck for the angles (modulo 2*PI)
!/*-----------------------------------------------------------------*/
      subroutine  checkangout(n, PV, PVcheck,seuil,jd0, dt,center,        &
     &                       vtarget, order, res, unit)
           implicit none
           integer, intent(in) :: n, center,vtarget, order
           integer, intent(inout) :: res
           real(8), intent(in) :: PV(12), PVcheck(12)
           real(8), intent(in) :: seuil, jd0, dt
           character(len=*), intent(in) :: unit
           integer j
           real(8) twopi, modpv
           
           twopi = ATAN2(1.D0, 1.D0)*4.D0;

               do j=1, n
                 modpv = MODULO(PV(j), twopi)-MODULO(PVcheck(j), twopi)
                 if (abs(modpv)>=seuil) then
                  write(*,*) "failure for ",vtarget, center
                  write(*,*) "at time ", jd0, dt
                  write(*,*) "unit=",unit
                  write(*,*) "order=",order
                  write(*,*) "diff=",PVcheck(j)-PV(j),PVcheck(j),PV(j)
                  write(*,*) "diff mod2pi=",modpv
                  write(*,*) "PVcheck mod2pi=",MODULO(PVcheck(j),twopi)
                  write(*,*) "PV mod2pi=",MODULO(PV(j), twopi)
                  write(*,*) PVcheck
                  write(*,*) PV
                  res = 2               
                 endif
                enddo
             
           return
           end 
                

!/*-----------------------------------------------------------------*/
!/* main program */
!/*-----------------------------------------------------------------*/
       program f2003mcomputeorder
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res, res2
           integer n, target1, center, status, j
           integer, parameter :: fileref =12
           integer unit
           integer order
           real(8) seuil
           real(8) jd0
           real(8) dt
           real(8) AU
           real(8) PV(12), PVcheck(12), PVcheck2(12)
           TYPE(C_PTR) :: peph
           include 'fopenfiles.h'
           
           res = 0
           
! open refernce data file
           open(fileref, file=trim(TOPSRCDIR)                            &
     &      //"example1_tests.dat",status="old")
           
! open the ephemeris file 
           peph=calceph_open(trim(TOPSRCDIR)                             &
     &      //"../examples/example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
             
! read the AU
               res2 = calceph_getconstant(peph,"AU"//C_NULL_CHAR,AU)
               if (res2.eq.0) then
                res = 2
               endif

             status = 0
             do while((status.eq.0).and.(res.eq.0))
! read the reference filejd0, target, center, 
              read(fileref, *,iostat=status) jd0, target1, center,        &
     &              (PVcheck2(j),j=1,6)
              if (status.eq.0) then
              
               dt = jd0-int(jd0)
               jd0 = int(jd0)+2.4515450000000000000D+06
! compute  the coordinates 
               res2 = calceph_compute(peph,jd0, dt, target1,center,PV)
               if (res2.eq.0) then
                res = 2
               endif
               
! loop on the order
               do order = 0, 1
               
! check the coordinates               
               seuil=3D-15
               n=3*(order+1)
               if ((target1.ne.15).and.(target1.ne.16)) then
                
!  check UA/DAY 
               seuil=3D-10
                unit = CALCEPH_UNIT_AU+CALCEPH_UNIT_DAY
                PVcheck=0
                res2 = calceph_compute_order(peph,jd0, dt, target1,      &
     &               center,unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
                call checkoutput(n,PVcheck2,PVcheck,seuil,jd0,dt,center, &
     &               target1,order, res,"AU/DAY")
               seuil=3D-15

!  check UA/SEC 
                unit = CALCEPH_UNIT_AU+CALCEPH_UNIT_SEC
                PVcheck=0
                res2 = calceph_compute_order(peph,jd0, dt, target1,      &
     &               center,unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
               PVcheck(4:6) = PVcheck(4:6)*86400D0
                call checkoutput(n,PV,PVcheck,seuil,jd0, dt,center,      &
     &               target1,order, res,"AU/SEC")
                
               seuil=3D-14
!  check KM/DAY 
                unit = CALCEPH_UNIT_KM+CALCEPH_UNIT_DAY
                PVcheck=0
                res2 = calceph_compute_order(peph,jd0, dt, target1,      &
     &               center,unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
               PVcheck = PVcheck/AU
                call checkoutput(n,PV,PVcheck,seuil,jd0, dt,center,      &
     &               target1,order, res,"KM/DAY")
                
!  check KM/SEC 
                unit = CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC
                PVcheck=0
                res2 = calceph_compute_order(peph,jd0, dt, target1,      &
     &               center,unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
               PVcheck = PVcheck/AU
               PVcheck(4:6) = PVcheck(4:6)*86400D0
                call checkoutput(n,PV,PVcheck,seuil,jd0, dt,center,      &
     &               target1,order, res, "KM/SEC")
              endif
              
               if (target1.eq.15) then
 !  check RAD/DAY 
                seuil=3D-12*10**(3*(order+1))
                unit = CALCEPH_UNIT_RAD+CALCEPH_UNIT_DAY
                PVcheck=0
                res2 = calceph_orient_order(peph,jd0, dt, target1,       &
     &               unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
                call checkangout(n,PVcheck2,PVcheck,seuil,jd0,dt,center, &
     &               target1,order, res,"RAD/DAY")
                
!  check RAD/SEC 
                unit = CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC
                PVcheck=0
                res2 = calceph_orient_order(peph,jd0, dt, target1,      &
     &               unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
               PVcheck(4:6) = PVcheck(4:6)*86400D0
                call checkangout(n,PVcheck2,PVcheck,seuil,jd0,dt,center, &
     &               target1, order, res,"RAD/SEC")
              endif

               seuil=3D-15

               if (target1.eq.16) then
 !  check SEC 
                unit = CALCEPH_UNIT_SEC
                PVcheck=0
                res2 = calceph_compute_order(peph,jd0, dt, target1,      &
     &               center,unit, order, PVcheck)
               if (res2.eq.0) then
                res = 2
               endif
                call checkoutput(n,PV,PVcheck,seuil,jd0, dt,center,      &
     &               target1,order, res,"SEC")
                
              endif


              enddo
              
              endif
             enddo
             
! close the ephemeris file 
             call calceph_close(peph)
            else
               write (*,*) "The ephemeris can't be opened"
               res = 2
            endif
            close(fileref)
       if (res.eq.0) then
        stop
       else
        stop 2
       endif
       end
      