!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003writephspk9parcheck.f 
!  \brief Check if writeph_spk9_* works with fortran 2003 compiler.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2026, CNRS
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
       program f2003writeph_spk9
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph

           use f2003writephinterpolationcheck
           implicit none
           TYPE(C_PTR) :: weph, reph
           integer ret, j, k
           integer fileref
           DATA fileref /12/
           real(8), dimension(1:4056*6) :: states
           real(8), dimension(1:4056) :: date
           integer itarget, icenter, iframe, iseg
           real(C_DOUBLE) firsttime, lasttime
           integer reserv
           integer, dimension(1:2) :: targets, record_counts
           real(C_DOUBLE), dimension(1:2) :: intlens
           character(len=40), dimension (2) :: segids

           include "fopenfiles.h"

           targets(1) = 199
           targets(2) = 299
           record_counts(:) = 4056
           segids(1) = "seg1"//C_NULL_CHAR 
           segids(2) = "seg2"//C_NULL_CHAR 


! open reference data file
           open(fileref, file=trim(TOPSRCDIR)                            &
     &      //"writephrefcoordinates.dat",status="old")
           do j=0, 4055
             read(fileref, *) date(j+1), (states(6*j+k),k=1,6)
           enddo
           close(fileref)    

! open the ephemeris file 
           weph=writeph_spk_create("f2003wephparspk9.bsp"//C_NULL_CHAR,          &
     &      "mykernel"//C_NULL_CHAR,0)
           if (.not.C_ASSOCIATED(weph)) then
            write (*,*) " failure on writeph_spk_create - dat"
            stop 2
           endif

           ret = writeph_comment(weph, "my comment"//C_NULL_CHAR)
           if (ret.eq.0) then
            write (*,*) " failure on writeph_comment"
            stop 2
           endif

           reserv = writeph_spk9_par_reserve(weph,2,targets, 10,1,              &
     &     date(1),0D0, date(4056), 0D0, record_counts, 9, segids)
           if (reserv.eq.0) then
            write (*,*) " failure on writeph_spk2_par_reserve"
            stop 2
           endif
           ret = writeph_spk9_par_write(weph,reserv, 0,0,                       &
     &     record_counts(1), states, date)
           if (ret.eq.0) then
            write (*,*) " failure on writeph_spk2_par_write"
            stop 2
           endif
           ret = writeph_spk9_par_write(weph,reserv, 1,0,                       &
     &     record_counts(1), states, date)
           if (ret.eq.0) then
            write (*,*) " failure on writeph_spk2_par_write"
            stop 2
           endif

           ret = writeph_close(weph)
           if (ret.eq.0) then
            write (*,*) " failure on writeph_close"
            stop 2
           endif

           reph = calceph_open("f2003wephparspk9.bsp"//C_NULL_CHAR)
           if (.not.C_ASSOCIATED(reph)) then
            write (*,*) " failure on calceph_open"
            stop 2
           endif

           if (calceph_getpositionrecordcount(reph).ne.2) then
            write (*,*) " failure on calceph_getpositionrecordcount"
            stop 2
           endif            
           ret = calceph_getpositionrecordindex2(reph,1,itarget,              &
     &       icenter, firsttime, lasttime, iframe, iseg)
           if (ret.eq.0) then
            write (*,*) " failure on calceph_getpositionrecordindex2"
            stop 2
           endif
           if ((icenter.ne.10).or.(itarget.ne.199)) then
            write (*,*) " failure on center or target", itarget,               &
     &       icenter, firsttime, lasttime, iframe, iseg
            stop 2
           endif
           if ((firsttime.ne.2451545D0).or.                                    &
     &       (abs(lasttime-2.45190995D6).ge.1D-2)) then 
            write (*,*) " failure on date", itarget,                           &
     &       icenter, firsttime, lasttime, iframe, iseg
            stop 2
           endif
           if ((iframe.ne.1).or.(iseg.ne.CALCEPH_SEGTYPE_SPK_9)) then
            write (*,*) " failure on iseg or frame", itarget,                  &
     &       icenter, firsttime, lasttime, iframe, iseg
            stop 2
           endif

           ret = writeph_check_states(reph,                                    &
     &     "writephrefcoordinateslagrange.dat"//C_NULL_CHAR, 199, 10,          &
     &       1D-7)
           if (ret.eq.0) then
            write (*,*) " failure on writeph_check_states"
            stop 2
           endif

           call calceph_close(reph)

           stop 

       end
      