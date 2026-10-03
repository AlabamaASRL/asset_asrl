!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003parallel.f 
!  \brief Example of usage of the parallel access to the same object.
!         This example computes the mean distance between earth and moon.
!         You can adjust the number of threads using the enviromental 
!         variable OMP_NUM_THREADS.
!
!         This example should be the openmp flags of your compiler, 
!         suchs as  -fopenmp or -qopenmp
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2019,  CNRS
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
!/* main program */
!/*-----------------------------------------------------------------*/
       program f2003multiple
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           real(8) jd0
           real(8) dt
           real(8) PV(6)
           integer j, NPOINTS
           TYPE(C_PTR) :: peph
           integer cont, t
           real(8) jdfirst, jdlast
           real(8) d, sum
           integer threadsafe
           
           jd0 = 2442458
           dt = 0.001D0
           NPOINTS = 1000000
           
! open the ephemeris file 
           peph = calceph_open("example1.dat"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
              write (*,*) "The ephemeris is already opened"

! prefetch the data : it is required to enable later  parallel access
           res = calceph_prefetch(peph)
           threadsafe = calceph_isthreadsafe(peph)
           if (threadsafe.eq.1) then   
             write (*,*) "The multiple access to the ephemeris "               &
     &        //"descriptor is thread-safe"
             write (*,*) "The loop 'for' will be parallelized."
             write (*,*) "The number of used threads is determined by"         &
     &        //" OMP_NUM_THREADS"
           else
             write (*,*) "The multiple access to the ephemeris "               &
     &        //"descriptor is  NOT thread-safe"
             write (*,*)"The loop 'for' will be executed in sequential"
           endif  

!$OMP PARALLEL DO reduction (+:sum) private(PV) if (threadsafe.eq.1)   
           do j=0, NPOINTS-1
            res = calceph_compute_unit(peph, jd0, j*dt, NAIFID_MOON,           &
     &       NAIFID_EARTH,                                                     &
     &       CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC+CALCEPH_USE_NAIFID,              &
     &       PV);
             sum = sum + sqrt(PV(1)**2+PV(2)**2+PV(3)**2)
           enddo 
!$OMP END PARALLEL DO
 
           d = sum/NPOINTS
            write(*,*) "mean distance between the Earth and the Moon :"       &
     &       ,d, " Kilometers"
! close the ephemeris file 
               call calceph_close(peph)
               write (*,*) "The ephemeris is already closed"
            else
               write (*,*) "The ephemeris can't be opened"
            endif
       stop
       end
      