!/*-----------------------------------------------------------------*/
!/*! 
!  \file f77mgetconstant_vd.f 
!  \brief Check if calceph_getconstantvd works with fortran 77 compiler.
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
       program f77mgetconstant_vd
           implicit none
           include 'f90calceph.h'
           integer*8 peph
           double precision value(4)
           integer res
           include 'fopenfiles.h'
           
           value(:) = 0
           res=f90calceph_open(peph, trim(TOPSRCDIR)                     &
     &      //"checktpc_11627.tpc")
           if (res.eq.1) then
            if (f90calceph_getconstantvd(peph, "BODY000_GMLIST1",       &
     &         value, 1).eq.1) then
               if ((value(1).ne.699)) then 
                stop 2
               endif 
            else
                stop 3
            endif
            if (f90calceph_getconstantvd(peph, "BODY000_GMLIST2",       &
     &         value, 2).eq.2) then
               if ((value(1).ne.499).or.(value(2).ne.599)) then 
                stop 4
               endif 
            else
                stop 5
            endif
            if (f90calceph_getconstantvd(peph, "BODY000_GMLIST4",       &
     &         value, 4).eq.4) then
               if ((value(1).ne.199).or.(value(2).ne.299).or.            &
     &         (value(3).ne.301).or.(value(4).ne.399)) then 
                stop 6
               endif 
            else
                stop 7
            endif
           endif
       stop     
       end
      
