!/*-----------------------------------------------------------------*/
!/*! 
!  \file f77mtimeconversion.f 
!  \brief Check if calceph_time_... works with fortran 77 compiler.
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

! check the result for the calendar
       subroutine check_cal(r1,yc,moc, dc,hc,mic,sc, ye, moe, de, he,      &
     &      mie, se)
        implicit none
        integer r1, yc,moc, dc,hc,mic,ye, moe, de, he,  mie
        real*8 sc,se 

        if (r1.eq.1) then
            if (yc.ne.ye) then 
                r1 = 0
                write(*,*) 'wrong year'
            endif
            if (moc.ne.moe) then 
                r1 = 0
                write(*,*) 'wrong month'
            endif
            if (dc.ne.de) then 
                r1 = 0
                write(*,*) 'wrong day'
            endif
            if (hc.ne.he) then 
                r1 = 0
                write(*,*) 'wrong hour'
            endif
            if (mic.ne.mie) then 
                r1 = 0
                write(*,*) 'wrong minute'
            endif
            if (abs(sc-se).ge.4D-5) then 
                r1 = 0
                write(*,*) 'wrong second'
            endif
            if (r1.eq.0) then 
              write(*,*) 'expected date: ', ye,moe,de,he,mie,se
              write(*,*) 'computed date: ', yc,moc,dc,hc,mic,sc
            endif
        else
            write(*,*) 'r1 invalid  : ',r1
            write(*,*) ' date: ', yc,moc,dc,hc,mic,sc
            write(*,*) 'expected date: ', ye,moe,de,he,mie,se
            write(*,*) 'computed date: ', yc,moc,dc,hc,mic,sc
        endif
       end


! check the result for the julianday
       subroutine check_jd(r1,jdic, jdfc, jdie, jdfe)
        implicit none
        real*8 jdic, jdfc, jdie, jdfe
        integer r1

        if (r1.eq.1) then
            if (abs(jdic-jdie).ge.1D-9) then 
                r1 = 0
                write(*,*) 'wrong integral part'
            endif
            if (abs(jdfc-jdfe).ge.1D-9) then 
                r1 = 0
                write(*,*) 'wrong fractional part'
            endif
            if (r1.eq.0) then 
              write(*,*) 'expected date: ', jdie, jdfe
              write(*,*) 'computed date: ', jdic, jdfc
            endif
        else
            write(*,*) 'r1 invalid  : ',r1
            write(*,*) ' date: ', jdie, jdfe
            write(*,*) 'expected date: ', jdie, jdfe
            write(*,*) 'computed date: ', jdic, jdfc
        endif
       end


!/*-----------------------------------------------------------------*/
!/* main program */
!/*-----------------------------------------------------------------*/
       program f77timeconversion
           implicit none
           include 'f90calceph.h'
           integer*8 peph
           integer res, res1
           integer r1, yy1, mo1, day1, hh1, mi1
           integer r2, yy2, mo2, day2, hh2, mi2
           integer r3, yy3, mo3, day3, hh3, mi3
           real*8 sec1, sec2, sec3
           integer r4, r5, r6,r7, r8, r9, r10, r11,r12, r13, r14, r15
           integer r16, r17, r18,r19
           real*8 jdi4, jdf4, jdi5, jdf5, jdi6, jdf6, jdi7, jdf7
           real*8 jdi9, jdf9, jdi10, jdf10, jdi11, jdf11, jdi12,jdf12
           real*8 jdi13,jdf13, jdi16, jdf16, jdi17, jdf17
           real*8 jdi18, jdf18
           character*1024 s19
           character*1024 filear(2)

           integer sumr
           include 'fopenfiles.h'
           
           filear(1) = trim(TOPSRCDIR)//"example_lsk.tls"
           filear(2) = trim(TOPSRCDIR)//"example_sclk.tsc"
           res = f90calceph_open_array(peph, 2, filear, 1024)
           if (res.eq.1) then
           
        write(*,*) "checking f90calceph_time_jd_to_cal ...."  
         
        r1=f90calceph_time_jd_to_cal(peph, CALCEPH_TAI, 2457948.0D0,     &
     &            0.76207291667D0, yy1, mo1, day1, hh1, mi1, sec1)
        call check_cal(r1,yy1,mo1,day1,hh1,mi1,sec1,2017,7,14,6,17,      &
     &  23.1D0)

        r2=f90calceph_time_jd_to_cal(peph, CALCEPH_TDB, 2457436.0D0,     &
     &            0.11835879629D0, yy2, mo2, day2, hh2, mi2, sec2)
        call check_cal(r2,yy2,mo2,day2,hh2,mi2,sec2,                     &
     &  2016, 2, 17, 14, 50, 26.2D0)

        r3=f90calceph_time_jd_to_cal(peph, CALCEPH_UTC, 2451545.0D0,     &
     &     0.04554398148148146D0, yy3, mo3, day3, hh3, mi3, sec3)
        call check_cal(r3,yy3,mo3,day3,hh3,mi3,sec3,                     &
     &  2000, 1, 1, 13, 05, 35.0D0)

        write(*,*) "checking f90calceph_time_cal_to_jd ...."  
         
        r4=f90calceph_time_cal_to_jd(peph, CALCEPH_TT,                   &
     &     2016, 2, 18, 14, 50, 26.2D0, jdi4, jdf4)
        call check_jd(r4,jdi4, jdf4,  2457437.0D0,0.11835879629D0)
 
         r5=f90calceph_time_cal_to_jd(peph, CALCEPH_UTC,                 &
     &     2000, 1, 12, 13, 05, 35.0D0, jdi5, jdf5)
        call check_jd(r5,jdi5, jdf5,2451556.0D0, 0.04554398148148146D0)

        write(*,*) "checking f90calceph_time_str_to_jd ...." 
          
         r6=f90calceph_time_str_to_jd(peph, CALCEPH_TT,                  &
     &     "2029 OCT 16 13:14:11.4967158436775208", jdi6, jdf6)
        call check_jd(r6,jdi6, jdf6,2462426.0D0,0.0515219527296722D0)

         r7=f90calceph_time_str_to_jd(peph, CALCEPH_TIMESCALE_FROM_STR,  &
     &     "1996 January 1,  06:00:0.0  (UTC)", jdi7, jdf7)
        call check_jd(r7,jdi7, jdf7,2450083D0, 0.75D0)

        write(*,*) "checking f90calceph_time_jd_tt_to_jd_tdb ...." 
         r8=f90calceph_time_set_relationship_tt_tdb(peph, 1)

         r9=f90calceph_time_jd_tt_to_jd_tdb(peph,                        &
     &     2452425D0,   0.989772840403D0, jdi9, jdf9)
        call check_jd(r9,jdi9, jdf9, 2452425D0,   0.989772851113D0)

        write(*,*) "checking f90calceph_time_jd_tdb_to_jd_tt ...." 
         r10=f90calceph_time_jd_tdb_to_jd_tt(peph,                       &
     &     2452425D0,   0.989772851113D0, jdi10, jdf10)
        call check_jd(r10,jdi10, jdf10,2452425D0, 0.989772840403D0)

         r16=f90calceph_time_jd_tcb_to_jd_tdb(peph,                        &
     &     2450083D0,0.0212302692234516D0, jdi16, jdf16)
        call check_jd(r16,jdi16, jdf16,2450083D0,0.0211226851679385D0)

        write(*,*) "checking f90calceph_time_jd_tdb_to_jd_tcb ...." 
         r17=f90calceph_time_jd_tdb_to_jd_tcb(peph,                       &
     &     2450083.0D0, 0.0211226851679385D0, jdi17, jdf17)
        call check_jd(r17,jdi17, jdf17,2450083D0,0.0212302692234516D0)
  
        write(*,*) "checking f90calceph_time_str_any_to_jd_tdb ...." 
         r11=f90calceph_time_str_any_to_jd_tdb(peph,                     &
     &     "1995 June 13  23:59:59.5  (UTC)", jdi11, jdf11)
        call check_jd(r11,jdi11,jdf11,2449882D0,0.5007023680955172D0)

        write(*,*) "checking f90calceph_time_str_utc_to_jd_tdb ...." 
         r12=f90calceph_time_str_utc_to_jd_tdb(peph,                     &
     &     "1988 June 13, 12:29:48 ", jdi12, jdf12)
        call check_jd(r12,jdi12,jdf12,2447326D0,0.0213447287678719D0)

        write(*,*) "checking f90calceph_time_cal_utc_to_jd_tdb ...." 
        r13=f90calceph_time_cal_utc_to_jd_tdb(peph, 1995, 12, 31,23,     &
     &       59,59.5D0, jdi13, jdf13)
        call check_jd(r13,jdi13,jdf13,2450083D0, 0.5007023601792753D0)

        write(*,*) "checking f90calceph_time_jd_tdb_to_cal_utc ...." 
        r14=f90calceph_time_jd_tdb_to_cal_utc(peph, 2450083D0,           &
     &   0.5007023601792753D0, yy1, mo1, day1, hh1, mi1, sec1 )
        call check_cal(r14,yy1,mo1,day1,hh1,mi1,sec1,1995, 12, 31,23,    &
     &       59,59.5D0)
        r15=f90calceph_time_jd_tdb_to_cal_utc(peph, 2450083D0,           &
     &   0.5007023601792753D0+1./86401., yy1,mo1,day1,hh1,mi1,sec1)
        call check_cal(r15,yy1,mo1,day1,hh1,mi1,sec1,1995, 12, 31,23,    &
     &       59,60.5D0)

        write(*,*) "checking f90..str_spacecraft_clock_to_jd_tdb ...." 
        r18 =  f90calceph_time_str_spacecraft_clock_to_jd_tdb(peph,28,   &
     &       "1/0706865508:31865", jdi18, jdf18)
        call check_jd(r18, jdi18, jdf18,2459725D0,                       &
     &      0.81458599643883644603D0)
         
        write(*,*) "checking f90..jd_tdb_to_str_spacecraft_clock ...." 
        r19 =  f90calceph_time_jd_tdb_to_str_spacecraft_clock(peph,28,   &
     &       2459744D0,  0.35689482379893888719D0, s19)
        if(trim(s19).ne."1/0708467563:63527") then
         r19 = 0
         write (*,*) "invalid jd_tdb_to_str_spacecraft_clock"
        endif

        call f90calceph_close(peph)       

        sumr = r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15+r16    &
     &       +r17+r18+r19
        if(sumr.eq.19)then
            stop
        endif
           endif
       write(*,*) 'stop with a failure'
       stop 2    
       end
      