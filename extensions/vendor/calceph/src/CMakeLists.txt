#/*-----------------------------------------------------------------*/
#  build chain for the source of the library.
#
#  Copyright, 2021-2026, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#
#/*-----------------------------------------------------------------*/
#/*-----------------------------------------------------------------*/
#/* License  of this file :
#  This file is "triple-licensed", you have to choose one  of the three licenses 
#  below to apply on this file.
#  
#     CeCILL-C
#     	The CeCILL-C license is close to the GNU LGPL.
#     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#   
#    or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#  
#    or CeCILL v2.1
#        The CeCILL license is compatible with the GNU GPL.
#        ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
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

CONFIGURE_FILE(calcephconfig.cmake.in calcephconfig.h)
CONFIGURE_FILE(f90calceph.h.in f90calceph.h  @ONLY)
CONFIGURE_FILE(f2003calceph.f.in f2003calceph.f )

if (ENABLE_FORTRAN)
    list (APPEND F2003_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/f2003calceph.f" f2003main.f)
else()
    set (F2003_SOURCES)
endif()    

if (ENABLE_FORTRAN AND CMAKE_GENERATOR MATCHES "^Visual Studio.*$")
    # a dummy library is required to produce object of the fortran files 
    # because Visual studio generators does not know how to handle Fortran files
    add_library(f2003calcephdummy OBJECT ${F2003_SOURCES})
    add_library(calceph ${CALCEPH_C_SOURCE} $<TARGET_OBJECTS:f2003calcephdummy>)
else()
    add_library(calceph ${CALCEPH_C_SOURCE} ${F2003_SOURCES})
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(calceph PROPERTIES VERSION ${CALCEPH_SOVERSION_FULL} SOVERSION ${CALCEPH_SOVERSION_COMPAT})
endif()

if(ENABLE_THREAD)
    if (Threads_FOUND)
        target_link_libraries(calceph PUBLIC Threads::Threads)
    endif()    
endif()

#####################################################################
# static compilation : PIE for the static library ?
#####################################################################
if (NOT BUILD_SHARED_LIBS)
    if (ENABLE_FORTRAN)
        if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.14.0")
            cmake_policy(SET CMP0083 NEW)
            include(CheckPIESupported)
            check_pie_supported(OUTPUT_VARIABLE piesupport LANGUAGES C)
            if (CMAKE_C_LINK_PIE_SUPPORTED)
                set(CALCEPH_USE_PIC TRUE CACHE INTERNAL "")
            endif()
        else()
            set(CALCEPH_USE_PIC TRUE CACHE INTERNAL "")
        endif()
    endif()
    if (ENABLE_PYTHON OR ENABLE_MEX_OCTAVE)
         set(CALCEPH_USE_PIC TRUE CACHE INTERNAL "")
    endif()
endif()

if (CALCEPH_USE_PIC)
    message(STATUS "Compiling with Position Independent Code (PIE) set to true")
    set_property(TARGET calceph PROPERTY POSITION_INDEPENDENT_CODE TRUE)
endif()


#####################################################################
# installation step
#####################################################################
if (CALCEPH_INSTALL)
    install(FILES calceph.h COMPONENT dev DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    if (ENABLE_FORTRAN)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/f90calceph.h" COMPONENT dev DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/calceph.mod" COMPONENT dev DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    endif()
    install(TARGETS  calceph EXPORT calcephTargets ARCHIVE COMPONENT dev  DESTINATION "${CMAKE_INSTALL_LIBDIR}" LIBRARY COMPONENT lib  DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    if (WIN32)
	    if (BUILD_SHARED_LIBS)
	        install(TARGETS calceph RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
	    endif()
    endif()
    target_include_directories(calceph PUBLIC $<INSTALL_INTERFACE:include/> )
endif()
target_include_directories(calceph PUBLIC $<BUILD_INTERFACE:>  )
target_include_directories(calceph INTERFACE $<INSTALL_INTERFACE:include/> )
target_include_directories(calceph INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>  )

