!/*-----------------------------------------------------------------*/
!/*! 
!  \file f2003mgetnamebyidss.f 
!  \brief Check if calceph_getnamebyidss works with fortran 2003 compiler.
!
!  \author  M. Gastineau 
!           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
!
!   Copyright, 2024, CNRS
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
!/* check the invalid id */
!/*-----------------------------------------------------------------*/
      function check_failure(peph, id, unit)
        USE, INTRINSIC :: ISO_C_BINDING
        use calceph
        implicit none
        TYPE(C_PTR), intent(in) :: peph
        character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) name
        integer unit
        integer ret
        integer id
        integer check_failure
        
        ret = calceph_getnamebyidss(peph, id, unit, name)

        if ((ret.ne.0)) then
            write(*,*) "FAIL: findvalid id but the name is invalid !"
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
      function check_valid(peph, name_expected, unit,id) 
        USE, INTRINSIC :: ISO_C_BINDING
        use calceph
        implicit none
        TYPE(C_PTR), intent(in) :: peph
        !character(len=*)  name_expected   
        CHARACTER(len=*), intent(in) ::name_expected
        character(len=CALCEPH_MAX_CONSTANTVALUE, kind=C_CHAR) name_c   
        character(len=CALCEPH_MAX_CONSTANTVALUE) name   
        integer unit
        integer ret
        integer id
        integer check_valid
        
        ret = calceph_getnamebyidss(peph, id, unit, name_c)   
        name = name_c(1:index(name_c, C_NULL_CHAR)-1)

        if ((ret.ne.1).or.(trim(name_expected).ne.trim(name))) then
            write(*,*) "FAIL: find invalid name but the id is valid!"
            write(*,*) 'name_c=',trim(name_c)
            write(*,*) 'name=',trim(name)
            write(*,*) 'unit=',unit
            write(*,*) 'id=',id
            write(*,*) 'name_expected=',trim(name_expected)
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
      program f2003mgetnamebyidss
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
        res = res +check_failure(peph, -5000, 0)
        res = res +check_failure(peph, 998, 0)
        res = res +check_failure(peph, 21, 0)
        res = res +check_failure(peph, -5000, CALCEPH_USE_NAIFID)
        res = res +check_failure(peph, 998, CALCEPH_USE_NAIFID)

        res = res +check_valid(peph, "JUPITER BARYCENTER", 0, 5)


        res = res +check_valid(peph, "SATURN BARYCENTER",                             &
     & CALCEPH_USE_NAIFID, 6)
      

        res = res +check_valid(peph, "SATURN",  CALCEPH_USE_NAIFID, 699)

        res = res +check_valid(peph, "MERCURY BARYCENTER", 0, 1)
        res = res +check_valid(peph, "VENUS BARYCENTER", 0, 2)
        res = res +check_valid(peph, "EARTH", 0, 3)
        res = res +check_valid(peph, "MARS BARYCENTER", 0, 4)
        res = res +check_valid(peph, "SATURN BARYCENTER", 0, 6)
        res = res +check_valid(peph, "URANUS BARYCENTER", 0, 7)
        res = res +check_valid(peph, "NEPTUNE BARYCENTER", 0, 8)
        res = res +check_valid(peph, "PLUTO BARYCENTER", 0, 9)
        res = res +check_valid(peph, "MOON", 0, 10)
        res = res +check_valid(peph, "SUN", 0, 11)
        res = res +check_valid(peph, "SOLAR SYSTEM BARYCENTER", 0,                    &
     & 12)
        res = res +check_valid(peph, "EARTH MOON BARYCENTER", 0,                      &
     & 13 )

        res = res +check_valid(peph, "EARTH",CALCEPH_USE_NAIFID,399)
        res = res +check_valid(peph, "MOON",CALCEPH_USE_NAIFID,301)
        res = res +check_valid(peph, "SUN", CALCEPH_USE_NAIFID, 10)
        res = res +check_valid(peph, "SOLAR SYSTEM BARYCENTER",                       &
     & CALCEPH_USE_NAIFID, 0)
        res = res +check_valid(peph, "EARTH MOON BARYCENTER",                         &
     & CALCEPH_USE_NAIFID, 3)

        res = res +check_valid(peph, "2 PALLAS", CALCEPH_USE_NAIFID,                  &
     & 2000002 )
        res = res +check_valid(peph, "1 CERES",                                       &
     & CALCEPH_USE_NAIFID,2000001)

          if (res.ne.0) then
            stop 4
          endif  

       call calceph_close(peph)

        endif

        stop     
      end
    