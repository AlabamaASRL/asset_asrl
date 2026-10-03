!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003mgetconstant_vs.f 
!  \brief Check if calceph_getconstantvs works with fortran 2003 compiler.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2008-2018, CNRS
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
       program f2003mgetconstant
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           TYPE(C_PTR) :: peph
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR)         &
     &       value(4)
           character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR)         &
     &       avalue
           include 'fopenfiles.h'
           
           peph=calceph_open(trim(TOPSRCDIR)                             &
     &      //"checktpc_str.tpc"//C_NULL_CHAR)
           if (C_ASSOCIATED(peph)) then
            if (calceph_getconstantss(peph, "DISTANCE_UNITS"            &
     &         //C_NULL_CHAR,avalue).eq.1) then
              if (trim(avalue).ne.'KILOMETERS'//C_NULL_CHAR) then 
                write(*,*) trim(avalue)
                stop 2
               endif 
            else
                stop 3
            endif
            if (calceph_getconstantvs(peph, "DISTANCE_UNITS"            &
     &         //C_NULL_CHAR,value, 1).eq.1) then
               if (trim(value(1)).ne.'KILOMETERS'//C_NULL_CHAR) then 
                write(*,*) trim(value(1))
                stop 9
               endif 
            else
                stop 10
            endif
            if (calceph_getconstantvs(peph, "MISSION_UNITS"             &
     &         //C_NULL_CHAR,value, 3).eq.3) then
               if ((trim(value(1)).ne.'KILOMETERS'//C_NULL_CHAR)         &
     &         .or.(trim(value(2)).ne.'SECONDS'//C_NULL_CHAR)            &
     &         .or.(trim(value(3)).ne.'KILOMETERS/SECOND'//C_NULL_CHAR)) &
     &         then 
                 write(*,*) trim(value(1))
                 write(*,*) trim(value(2))
                 write(*,*) trim(value(3))
               stop 4
               endif 
            else
                stop 5
            endif
         endif   
       stop     
       end
      