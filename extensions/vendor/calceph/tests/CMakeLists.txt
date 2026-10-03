#/*-----------------------------------------------------------------*/
#/*! 
#  \file CMakeLists.txt
#  \brief build chain of the tests for cmake. 
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2021-2026, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#  
#*/
#/*-----------------------------------------------------------------*/
#
#/*-----------------------------------------------------------------*/
#/* License  of this file :
#  This file is "triple-licensed", you have to choose one  of the three licenses 
#  below to apply on this file.
#  
#     CeCILL-C
#     	The CeCILL-C license is close to the GNU LGPL.
#     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#   
#  or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#  
#  or CeCILL v2.1
#       The CeCILL license is compatible with the GNU GPL.
#       ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
#  
# 
# This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under 
# French law and abiding by the rules of distribution of free software.  
# You can  use, modify and/ or redistribute the software under the terms 
# of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA  
# at the following URL "http://www.cecill.info". 
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
# */
# /*-----------------------------------------------------------------*/



include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../src")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(calcephtest countlines.c openfiles.c numericalderivative.c  writephcommoncheck.c)
target_link_libraries (calcephtest calceph ${LIBM_LIBRARIES})
if (CALCEPH_USE_PIC)
	set_property(TARGET calcephtest PROPERTY POSITION_INDEPENDENT_CODE TRUE)
endif()

file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/example1_tests_23_derivative.dat")
set(top_srcdir "${CMAKE_CURRENT_SOURCE_DIR}")
string(REPLACE "/" "/\"\n     & //\"" FORTRAN_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
CONFIGURE_FILE(fopenfiles.h.in fopenfiles.h )
CONFIGURE_FILE(fthreadsafe.h.in fthreadsafe.h )

#####################################################################
# common options for test files
function(buildtestscommon execname)
  add_test( ${execname}  ${execname} )
  set_tests_properties(${execname} PROPERTIES ENVIRONMENT "srcdir=${CMAKE_CURRENT_SOURCE_DIR}")  
  if (WIN32)  
	set_property(TEST ${execname} PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_append:${CMAKE_BINARY_DIR}/src")
  endif()
endfunction(buildtestscommon)

#####################################################################
# single C test file
function(buildtestscsimple execname sourcename)
  add_executable (${execname} ${sourcename})
  target_link_libraries (${execname} PRIVATE calcephtest calceph ${LIBM_LIBRARIES})
  buildtestscommon( ${execname} )
 endfunction(buildtestscsimple)

set (LIST_TEST_C_SIMPLE cversion csopenfail cmopenfail  csopen cmopen cmgetconstant csgetconstant cscheck cmcheck 
                 csgetconstantindex cmgetconstantindex
                 cmgetconstanttxtpck cmgetconstantindextxtpck cmcheckspk cmcomputeunitfail 
                 cmcomputeunit cmcomputeunit_naifid cmcheckspktt cmorientunitfail matrixfail 
                 tfmatrix cmcomputeunitfail2 checktpc_10086     
                 cmorientunitfailspk cmcomputeorder01 cmcomputeorder23 checkcst_11015 cmorientunitfail_11025 
                 crotangmomunitfail crotangmomunitfail2 crotangmomunit crotangmomorder0 crotangmomorder123 
                 cmtimescale cstimescale cmtimespan cstimespan cgetpositionrecordindex cgetorientrecordindex 
                 cmorientunitfail_11612 cmgetconstant_11627  cmopenfail_11628 cmgetconstantvd 
                 cmgetconstantvs csfileversionstr cmparallelcheck cmpolerade checktpc_12301 
                 cmaxsupportedderivative checktpc_12730 checktpc_incremental cmcomputeunit_framerelative
                 cmgetidbyname cmgetnamebyidss checktf_7 cmgetconstanttxtik
                 checkgetfov checkgetfovfail cmgetconstanttxtlsk
                 cmgetconstanttxtsclk checkparsetime timetestjdcal timetesttdbtt
                 timetestutctdb timetestanytdb
                 timetesttdbtcb timetesttdbspacecraft timetesttdbspacecraftfail
)
                 
foreach (C_SIMPLE IN LISTS LIST_TEST_C_SIMPLE)
    buildtestscsimple(${C_SIMPLE} ${C_SIMPLE}.c)
endforeach()



#####################################################################
# C test file with computehermite
function(buildtestschermite execname sourcename)
  add_executable (${execname} ${sourcename} cmcomputeunit_hermite.c)
  target_link_libraries (${execname} calcephtest calceph ${LIBM_LIBRARIES})
  buildtestscommon( ${execname} )
endfunction(buildtestschermite)

set (LIST_TEST_C_HERMITE cmcomputeunit_seg1 cmcomputeunit_seg8
                 cmcomputeunit_seg9 cmcomputeunit_seg12  cmcomputeunit_seg13 
                 cmcomputeunit_seg14 cmcomputeunit_seg21
                 cmcomputeunit_frame
)
                 
foreach (C_HERMITE IN LISTS LIST_TEST_C_HERMITE)
    buildtestschermite(${C_HERMITE} ${C_HERMITE}.c)
endforeach()


#####################################################################
# C test file with writeph
function(buildtestswritephcommon execname sourcename)
  add_executable (${execname} ${sourcename})
  target_link_libraries (${execname} calcephtest calceph ${LIBM_LIBRARIES})
  buildtestscommon( ${execname} )
endfunction(buildtestswritephcommon)

set (LIST_TEST_C_WRITEPHCOMMONCHECK
                writephspk2seqcheck writephspk2parcheck
                writephspk3seqcheck writephspk3parcheck
                writephspk8seqcheck writephspk8parcheck
                writephspk9seqcheck writephspk9parcheck
                writephspk12seqcheck writephspk12parcheck
                writephspk13seqcheck writephspk13parcheck
                writephspk14seqcheck
                writephspk102seqcheck writephspk102parcheck
                writephspk103seqcheck writephspk103parcheck
                writephpck2seqcheck writephpck2parcheck
                writephpck3seqcheck writephpck3parcheck
                writephpck102seqcheck writephpck102parcheck
                writephpck103seqcheck writephpck103parcheck
                writephdumprestoreseqcheck writephdumprestoreparcheck writephdumprestorefail
                writephcreateopencommentfail
)

foreach(C_WRITEPHCOMMONCHECK IN LISTS LIST_TEST_C_WRITEPHCOMMONCHECK)
    buildtestswritephcommon(${C_WRITEPHCOMMONCHECK} ${C_WRITEPHCOMMONCHECK}.c)
endforeach()


#####################################################################
# Fortran test file
if (ENABLE_FORTRAN)

    function(buildtestsfsimple execname sourcename)
      add_executable (${execname} ${sourcename})
      target_link_libraries (${execname} calcephtest calceph ${LIBM_LIBRARIES})
	    buildtestscommon( ${execname} )
    endfunction(buildtestsfsimple)

    set (LIST_TEST_FORTRAN_SIMPLE f2003sopenfail f2003mopenfail f2003sopen  f2003mopen f2003sgetconstant 
                     f2003mgetconstant f2003mcheck f2003sgetconstantindex f2003mgetconstantindex 
                     f2003mcomputeunit f2003mcomputeunit_naifid f2003version 
                     f2003mcomputeorder f2003rotangmomunit f2003rotangmomorder 
                     f2003mtimescale f2003mtimespan f2003stimescale f2003stimespan  
                     f2003getpositionrecordindex f2003getorientrecordindex 
                     f2003mgetconstantvd f2003mgetconstantvs f2003mfileversion f2003sfileversion 
                     f2003misthreadsafe f2003maxsupportedderivativeorder 
                     f2003mgetidbyname f2003mgetnamebyidss
                     f2003mtimeconversion f2003mgetfov
                     f77sopenfail f77mopenfail f77sopen  f77mopen f77sgetconstant 
                     f77mgetconstant f77mcheck f77sgetconstantindex f77mgetconstantindex 
                     f77mcomputeunit  f77mcomputeunit_naifid f77version 
                     f77mcomputeorder f77rotangmomunit f77rotangmomorder 
                     f77mtimescale f77mtimespan f77stimescale f77stimespan  
                     f77getpositionrecordindex f77getorientrecordindex 
                     f77mgetconstantvd f77mgetconstantvs f77mfileversion f77sfileversion 
                     f77misthreadsafe f77maxsupportedderivativeorder
                     f77mgetidbyname f77mgetnamebyidss
                     f77mtimeconversion f77mgetfov
    )
                 
    foreach( FORTRAN_SIMPLE IN LISTS LIST_TEST_FORTRAN_SIMPLE)
        buildtestsfsimple(${FORTRAN_SIMPLE} ${FORTRAN_SIMPLE}.f)
    endforeach()


    # writing tests
    add_library(f2003writephtest f2003writephinterpolationcheck.f)

   function(buildtestsfwrite execname sourcename)
      add_executable (${execname} ${sourcename})
      target_link_libraries (${execname} f2003writephtest calcephtest calceph ${LIBM_LIBRARIES})
	    buildtestscommon( ${execname} )
    endfunction(buildtestsfwrite)

    set (LIST_TEST_FORTRAN_WRITE f2003writephspk2seqcheck f2003writephspk3seqcheck
      f2003writephspk8seqcheck f2003writephspk9seqcheck
      f2003writephspk12seqcheck f2003writephspk13seqcheck
      f2003writephspk102seqcheck f2003writephspk103seqcheck
      f2003writephspk2parcheck f2003writephspk3parcheck
      f2003writephspk8parcheck f2003writephspk9parcheck
      f2003writephspk12parcheck f2003writephspk13parcheck
      f2003writephspk102parcheck f2003writephspk103parcheck
      f2003writephpck2seqcheck f2003writephpck2parcheck
      f2003writephpck3seqcheck f2003writephpck3parcheck
      f2003writephpck102seqcheck f2003writephpck102parcheck
      f2003writephpck103seqcheck f2003writephpck103parcheck      
    )
                 
    foreach( FORTRAN_WRITE IN LISTS LIST_TEST_FORTRAN_WRITE)
        buildtestsfwrite(${FORTRAN_WRITE} ${FORTRAN_WRITE}.f)
    endforeach()

    # clean the fortran produced bsp
    set (LIST_TEST_FORTRAN_PRODUCED_BSP f2003wephseqspk2.bsp f2003wephseqspk3.bsp
      f2003wephseqspk8.bsp f2003wephseqspk9.bsp
      f2003wephseqspk12.bsp f2003wephseqspk13.bsp
      f2003wephseqspk102.bsp f2003wephseqspk103.bsp
      f2003wephparspk2.bsp f2003wephparspk3.bsp
      f2003wephparspk8.bsp f2003wephparspk9.bsp
      f2003wephparspk12.bsp f2003wephparqspk13.bsp
      f2003wephparspk102.bsp f2003wephparspk103.bsp
      f2003wephseqpck2.bpc f2003wephseqpck3.bpc
      f2003wephseqpck102.bpc f2003wephseqpck103.bpc
      f2003wephparpck2.bpc f2003wephparpck3.bpc
      f2003wephparpck102.bpc f2003wephparpck103.bpc
      f2003wephparspk13.bsp 
    )
            
    foreach( PRODUCED_BSP IN LISTS LIST_TEST_FORTRAN_PRODUCED_BSP)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/${PRODUCED_BSP}" ) 
    endforeach()

endif()

#####################################################################

set_property(DIRECTORY APPEND
        PROPERTY ADDITIONAL_MAKE_CLEAN_FILES example1_tests_23_derivative.dat
) 

# clean the c produced bsp
set (LIST_TEST_C_PRODUCED_BSP writephpck102par.bsp              writephpck3par.bsp                writephspk12par.bsp               writephspk3.bsp
writephcreateopencommentfail.bsp  writephpck103.bsp                 writephspk102.bsp                 writephspk13.bsp                  writephspk3par.bsp
writephdumprestorepar.bsp         writephpck103par.bsp              writephspk102par.bsp              writephspk13par.bsp               writephspk8.bsp
writephdumprestoreseq.bsp         writephpck2.bsp                   writephspk103.bsp                 writephspk14.bsp                  writephspk8par.bsp
writephdumprestorfail.bsp         writephpck2par.bsp                writephspk103par.bsp              writephspk2.bsp                   writephspk9.bsp
writephpck102.bsp                 writephpck3.bsp                   writephspk12.bsp                  writephspk2par.bsp                writephspk9par.bsp)

    foreach( PRODUCED_BSP IN LISTS LIST_TEST_C_PRODUCED_BSP)
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/${PRODUCED_BSP}" ) 
    endforeach()

#####################################################################
# C test file with extra files
function(buildtestsextra execname sourcename)
  add_executable (${execname} ${sourcename} cmcomputeunit_hermite.c)
  target_link_libraries (${execname} calcephtest calceph ${LIBM_LIBRARIES})
endfunction(buildtestsextra)

set (LIST_TEST_C_EXTRA  cmcomputeunit_seg5  cmcomputeunit_seg17 
                 cmcomputeunit_seg18 cmcomputeunit_seg19 cmcomputeunit_seg20
)
                 
foreach (C_EXTRA IN LISTS LIST_TEST_C_EXTRA)
    buildtestsextra(${C_EXTRA} ${C_EXTRA}.c)
endforeach()
