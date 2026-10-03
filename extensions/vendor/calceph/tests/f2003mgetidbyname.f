!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003mgetidbyname.f 
!  \brief Check if calceph_getidbyname works with fortran 2003 compiler.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2023-2024, CNRS
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
!/* check the invalid name */
!/*-----------------------------------------------------------------*/
      function check_failure(peph, name, unit)
        USE, INTRINSIC :: ISO_C_BINDING
        use calceph
        implicit none
        TYPE(C_PTR), intent(in) :: peph
        character(len=*) name
        integer unit
        integer ret
        integer id
        integer check_failure
        
        id = -1

        ret = calceph_getidbyname(peph, trim(name)//C_NULL_CHAR, unit,   &
     &     id)

        if ((ret.ne.0)) then
            write(*,*) "FAIL: findvalid id but the name is invalid !!!"
            write(*,*) 'name=',name
            write(*,*) 'unit=',unit
            write(*,*) 'id=',id
            write(*,*) 'ret=',ret
        endif  
        
        check_failure = ret
      end function

!/*-----------------------------------------------------------------*/
!/* check the valid name */
!/*-----------------------------------------------------------------*/
      function check_valid(peph, name, unit,id_expected) 
        USE, INTRINSIC :: ISO_C_BINDING
        use calceph
        implicit none
        TYPE(C_PTR), intent(in) :: peph
        character(len=*) :: name
        integer unit
        integer id_expected
        integer ret
        integer id
        integer check_valid
        
        id = -1

        ret = calceph_getidbyname(peph, trim(name)//C_NULL_CHAR, unit,   &
     &     id)

        if ((ret.ne.1).or.(id_expected.ne.id)) then
            write(*,*) "FAIL: find invalid id but the name is valid !!!"
            write(*,*) 'name=',trim(name)
            write(*,*) 'unit=',unit
            write(*,*) 'id=',id
            write(*,*) 'id_expected=',id_expected
            write(*,*) 'ret=',ret
            ret = 1
        else
            ret = 0
        endif  
        
        check_valid = ret
      end function


!/*-----------------------------------------------------------------*/
!/* main program */
!/*-----------------------------------------------------------------*/
      program f2003mgetidbyname
        USE, INTRINSIC :: ISO_C_BINDING
        use calceph
        implicit none
        TYPE(C_PTR) :: peph
        integer res
        character(len=CALCEPH_MAX_CONSTANTVALUE) name
        external check_valid
        external check_failure
        integer check_failure
        integer check_valid

        include 'fopenfiles.h'
        
        peph = calceph_open(trim(TOPSRCDIR)                             &
     &      //"../examples/example1.tpc"//C_NULL_CHAR) 

        if (.not.C_ASSOCIATED(peph)) then

            res = 1
            stop 5
        else 

        res = 0
        res = res +check_failure(peph, "", 0)
        res = res +check_failure(peph, "JUPITERBARYCENTER", 0)
        res = res +check_failure(peph, "", CALCEPH_USE_NAIFID)
        res = res +check_failure(peph, "JUPITERBARYCENTER",              &
     & CALCEPH_USE_NAIFID)
        res = res +check_failure(peph, "XX", 0)
        res = res +check_failure(peph, "XX", CALCEPH_USE_NAIFID)

        res = res +check_valid(peph, "JUPITER BARYCENTER", 0, 5)
        res = res +check_valid(peph, "Jupiter Barycenter", 0, 5)
        res = res +check_valid(peph, "  Jupiter Barycenter", 0, 5)
        res = res +check_valid(peph, "Jupiter   Barycenter", 0, 5)
        res = res +check_valid(peph, " Jupiter Barycenter ", 0, 5)

        res = res +check_valid(peph, "SATURN BARYCENTER",               &
     & CALCEPH_USE_NAIFID, 6)
        res = res +check_valid(peph, "SaTURN Barycenter",               &
     & CALCEPH_USE_NAIFID, 6)
        res = res +check_valid(peph, "   Saturn Barycenter",            &
     & CALCEPH_USE_NAIFID, 6)
        res = res +check_valid(peph, "Saturn    Barycenter",            &
     & CALCEPH_USE_NAIFID, 6)
        res = res +check_valid(peph, "   Saturn Barycenter   ",         &
     & CALCEPH_USE_NAIFID, 6)

        res = res +check_valid(peph, "SATURN ",                         &
     &  CALCEPH_USE_NAIFID, 699)
        res = res +check_valid(peph, "SaTURN    ",                      &
     & CALCEPH_USE_NAIFID, 699)
        res = res +check_valid(peph, "   Saturn",                       &
     & CALCEPH_USE_NAIFID, 699) 
        res = res +check_valid(peph, "Saturn",                          &
     & CALCEPH_USE_NAIFID, 699)
        res = res +check_valid(peph, "   Saturn   ",                    &
     & CALCEPH_USE_NAIFID, 699)

        res = res +check_valid(peph, "Mercury Barycenter", 0, 1)
        res = res +check_valid(peph, "Venus Barycenter", 0, 2)
        res = res +check_valid(peph, " Earth ", 0, 3)
        res = res +check_valid(peph, "Mars Barycenter", 0, 4)
        res = res +check_valid(peph, "Saturn Barycenter", 0, 6)
        res = res +check_valid(peph, "Uranus Barycenter", 0, 7)
        res = res +check_valid(peph, "Neptune Barycenter", 0, 8)
        res = res +check_valid(peph, "Pluto Barycenter", 0, 9)
        res = res +check_valid(peph, "Moon", 0, 10)
        res = res +check_valid(peph, "Sun", 0, 11)
        res = res +check_valid(peph, "Solar   System   barycenter", 0,                &
     & 12)
        res = res +check_valid(peph, " Earth Moon   barycenter ", 0,                  &
     & 13 )

        res = res +check_valid(peph, " Earth ",CALCEPH_USE_NAIFID,399)
        res = res +check_valid(peph, " Moon ",CALCEPH_USE_NAIFID,301)
        res = res +check_valid(peph, "Sun", CALCEPH_USE_NAIFID, 10)
        res = res +check_valid(peph, "Solar   System   barycenter",                   &
     & CALCEPH_USE_NAIFID, 0)
        res = res +check_valid(peph, " Earth Moon   barycenter ",                     &
     & CALCEPH_USE_NAIFID, 3)

        res = res +check_valid(peph, "2 PALLAS", CALCEPH_USE_NAIFID,                  &
     & 2000002 )
        res = res +check_valid(peph, "1 CERES", CALCEPH_USE_NAIFID,                   &
     & 2000001)

          if (res.ne.0) then
            stop 4
          endif  

       call calceph_close(peph)

        endif

        stop     
      end
    