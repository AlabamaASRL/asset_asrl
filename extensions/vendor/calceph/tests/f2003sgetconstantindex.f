!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003sgetconstantindex.f 
!  \brief  Check if calceph_sgetconstantindex works with fortran 2003 compiler.
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
       program f2003sgetconstantindex
           USE, INTRINSIC :: ISO_C_BINDING
           use calceph
           implicit none
           integer res
           integer j
           integer countconst
           real(8) :: valueconstant
           character(len=CALCEPH_MAX_CONSTANTNAME) nameconstant
           include 'fopenfiles.h'
           
           res=calceph_sopen(trim(TOPSRCDIR)                            &
     &      //"../examples/example1.dat"//C_NULL_CHAR)
           if (res.eq.1) then
             countconst = calceph_sgetconstantcount()
             do j=1, countconst
                res = calceph_sgetconstantindex(j,nameconstant,          &
     &                                       valueconstant)
                if (res.eq.0) then
                    stop 2
                elseif (index(nameconstant,achar(0)).gt.0) then
                 write(*,*)'nul character in ',trim(nameconstant)
                 stop 2
                endif
             enddo
             if ((countconst .le. 4) .or. (countconst.gt.500)) then
                res = 0
             endif
             if (calceph_sgetconstantindex(0, nameconstant,             &
     &          valueconstant).ne.0) then
                res = 0
             endif
             if (calceph_sgetconstantindex(countconst+1,                &
     &          nameconstant, valueconstant).ne.0) then
                res = 0
             endif
             
           endif
           
           ! stop on sucess
           if (res.eq.1) then 
            stop
           endif
           ! stop on error
           stop 2
       end
