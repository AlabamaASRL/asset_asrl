#/*-----------------------------------------------------------------*/
#  complete build chain for cmake. 
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

cmake_minimum_required (VERSION 3.7.0...4.3.4)
project (calceph VERSION 5.0.1 LANGUAGES C)
set(CALCEPH_RELEASE_DATE_FULL "September 18, 2026")

# ABI version number 
# follow  rules : http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
set(CALCEPH_SOVERSION_COMPAT 2)
set(CALCEPH_SOVERSION_FULL ${CALCEPH_SOVERSION_COMPAT}.1.1)


#####################################################################
#  set the main variables
#####################################################################
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/toolchain")

set(CALCEPH_CONTACT_EMAIL "Mickael.Gastineau@obspm.fr")
set(CALCEPH_MAINTAINER_NAME "Mickaël Gastineau")
set(CALCEPH_URL "https://calceph.imcce.fr/")
set(CALCEPH_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CALCEPH_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CALCEPH_VERSION_PATCH ${PROJECT_VERSION_PATCH})


#####################################################################
#  set the options
#####################################################################
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    option(ENABLE_FORTRAN "enable fortran API" ON)
else()
    option(ENABLE_FORTRAN "enable fortran API" OFF) 
endif()
    
option(ENABLE_PYTHON "enable python API" OFF)
option(ENABLE_MEX_OCTAVE "enable mex API for Octave" OFF)
option(BUILD_SHARED_LIBS "enable the shared library, otherwise build the static library" OFF)
option(ENABLE_THREAD "enable thread-safe version for high-level API" ON)
option(CALCEPH_INSTALL "enable the generation of install target" ON)
option(ENABLE_DOC "enable the generation and installation of the html documentation" OFF)
option(ENABLE_PDF "enable the generation and installation of the pdf documentation" OFF)
option(ENABLE_SANITIZER "enable the sanitizer options (reserved for developers)" OFF)

#####################################################################
#  enable fortran if required
#####################################################################
if (ENABLE_FORTRAN)
    enable_language(Fortran)
    include(FortranCInterface)
    FortranCInterface_VERIFY(QUIET)
    FortranCInterface_HEADER(src/calcephconfigfortran.h MACRO_NAMESPACE "FC_")
    if("${FortranCInterface_GLOBAL_CASE}" STREQUAL "UPPER")
        set(CALCEPH_FORTRAN_GLOBALCASE "NAME") 
    endif()
else()
    CONFIGURE_FILE(src/calcephdisablefortran.cmake.in src/calcephconfigfortran.h  @ONLY)
endif()

#####################################################################
#  enable python if required
#####################################################################
if (ENABLE_PYTHON)
    if (CMAKE_VERSION VERSION_LESS "3.12.0")
        message(ERROR "Python API interface requires CMake 3.12 or later ")
    endif()
    include(FindPython)
    find_package (Python COMPONENTS Interpreter Development REQUIRED)
    file(READ README.pypi SETUPPY_LONG_DESCRIPTION)
    option(ENABLE_PYTHON_TESTS "enable python tests (developers only)" ON)
    if (BUILD_SHARED_LIBS)
        message(STATUS "Python API requires to build the static library")
        set(BUILD_SHARED_LIBS OFF)
    endif()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}>")
endif()


#####################################################################
#  enable mex if required
#####################################################################
if (ENABLE_MEX_OCTAVE)
    if (CMAKE_VERSION VERSION_LESS "3.12.0")
        essage(ERROR "Python API interface requires CMake 3.12 or later ")
    endif()
    include(FindOctave)
    find_package(Octave)
    if (BUILD_SHARED_LIBS)
        message(STATUS "MEX API requires to build the static library")
        set(BUILD_SHARED_LIBS OFF)
    endif()
endif()


#####################################################################
#  set the compiler flags 
#####################################################################
IF ("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  MESSAGE(STATUS "Add the option '-fp-model precise' for the Intel compilers")
  string(APPEND CMAKE_C_FLAGS " -fp-model precise ")
ENDIF ()
# Define for the function strtod_l 
if (NOT ANDROID)
string(APPEND CMAKE_C_FLAGS " -D_GNU_SOURCE=1 ")
endif()
# Define to 1 to make fseeko visible on some hosts (e.g. glibc 2.2).
string(APPEND CMAKE_C_FLAGS " -D_LARGEFILE_SOURCE=1 ")
string(APPEND CMAKE_C_FLAGS " -D_LARGE_FILES=1 ")
if(ENABLE_SANITIZER)
  MESSAGE(STATUS "Add the option '-fsanitize=address -g -O0' for the sanitizer")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -g -O0")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fsanitize=address -g -O0")
endif() 
if (WIN32)
  if (BUILD_SHARED_LIBS)
	  if (MINGW OR MSYS OR CYGWIN)
	  # only for the microsoft compiler
	  elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
		MESSAGE(STATUS "Exporting all symbols on windows")
		set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
	  endif()
  endif()
endif()

#####################################################################
#  detect the headers
#####################################################################
include(CheckIncludeFiles)
if(ENABLE_THREAD)
    find_package(Threads)
endif()
CHECK_INCLUDE_FILES(stdio.h HAVE_STDIO_H )
CHECK_INCLUDE_FILES(stdlib.h HAVE_STDLIB_H )
CHECK_INCLUDE_FILES(stdarg.h STDC_HEADERS)
CHECK_INCLUDE_FILES(limits.h HAVE_LIMITS_H)
CHECK_INCLUDE_FILES(string.h HAVE_STRING_H)
CHECK_INCLUDE_FILES(errno.h HAVE_ERRNO_H)
CHECK_INCLUDE_FILES(math.h HAVE_MATH_H)
CHECK_INCLUDE_FILES(stdbool.h HAVE_STDBOOL_H)
CHECK_INCLUDE_FILES(sys/stat.h HAVE_SYS_STAT_H)
CHECK_INCLUDE_FILES(sys/types.h HAVE_SYS_TYPES_H)
CHECK_INCLUDE_FILES(unistd.h HAVE_UNISTD_H)
CHECK_INCLUDE_FILES(time.h HAVE_TIME_H)
CHECK_INCLUDE_FILES(sys/mman.h HAVE_SYS_MMAN_H)
CHECK_INCLUDE_FILES(sys/resource.h HAVE_SYS_RESOURCE_H)
CHECK_INCLUDE_FILES(xlocale.h HAVE_XLOCALE_H)
CHECK_INCLUDE_FILES(locale.h HAVE_LOCALE_H)
if (CMAKE_USE_PTHREADS_INIT)
    CHECK_INCLUDE_FILES(pthread.h HAVE_PTHREAD_H)
    set(HAVE_PTHREAD 1)
    set(FORTRAN_HAVE_PTHREAD 1)
else()
    set(FORTRAN_HAVE_PTHREAD 0)
endif()
if (WIN32)
    set(HAVE_WINDOWS_H 1)
endif()

#####################################################################
#  detect the types
#####################################################################
include(CheckCSourceCompiles)
CHECK_C_SOURCE_COMPILES(
    "
    #include <stdbool.h>
    int main(int argc, char** argv) { (void)argc; (void)argv; bool b = true; return b!=false; }
    " 
    HAVE_BOOL
)

#####################################################################
#  detect the attributes
#####################################################################
CHECK_C_SOURCE_COMPILES(
    "
    #include <math.h>
    int main(int argc __attribute__((__unused__)) , char** argv __attribute__((__unused__))) {  return 0; }
    "  
    HAVE_VAR_ATTRIBUTE_UNUSED
)

#####################################################################
#  detect iso c11 threads
#####################################################################
include(CheckCSourceCompiles)
CHECK_C_SOURCE_COMPILES(
    "
    #include <threads.h>
    #include <stdio.h>
    #if __STDC_NO_THREADS__
    ...
    #endif
    static thread_local int *pephbin = NULL;
    int fn( void *  ) { return thrd_success; }
    int main(int argc, char** argv) { (void)argc; (void)argv; mtx_t mutex; thrd_t thread; thrd_create( &thread, fn, &argc); return 0; }
    " 
    HAVE_ISO_C11_THREAD
)


#####################################################################
#  detect the libraries
#####################################################################
# the math library is not required with some compilers, such as microsoft compilers or MinGW
include(CheckLibraryExists)
check_library_exists(m exp "" HAS_MATHLIB)
if(HAS_MATHLIB)
    set(LIBM_LIBRARIES m)
endif()

#####################################################################
#  detect the functions
#####################################################################
include (CheckFunctionExists)
CHECK_C_SOURCE_COMPILES(
    "
    #include <math.h>
    int main(int argc, char** argv) { (void)argc; (void)argv; return rint(1.0E0); }
    "  
    HAVE_RINT
)

CHECK_FUNCTION_EXISTS(snprintf HAVE_SNPRINTF)
CHECK_FUNCTION_EXISTS(vasprintf HAVE_VASPRINTF)
CHECK_FUNCTION_EXISTS(_vscprintf HAVE__VSCPRINTF)
CHECK_FUNCTION_EXISTS(vsnprintf HAVE_VSNPRINTF)
CHECK_FUNCTION_EXISTS(strncmp HAVE_STRNCMP)
CHECK_FUNCTION_EXISTS(strcasecmp HAVE_STRCASECMP)
CHECK_FUNCTION_EXISTS(_stricmp HAVE__STRICMP)
CHECK_FUNCTION_EXISTS(strtod_l HAVE_STRTOD_L)
CHECK_FUNCTION_EXISTS(_strtod_l HAVE__STRTOD_L)
CHECK_FUNCTION_EXISTS(newlocale HAVE_NEWLOCALE)
CHECK_FUNCTION_EXISTS(freelocale HAVE_FREELOCALE)
CHECK_FUNCTION_EXISTS(setlocale HAVE_SETLOCALE)
CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
CHECK_FUNCTION_EXISTS(fseeko HAVE_FSEEKO)
CHECK_FUNCTION_EXISTS(_fseeki64 HAVE__FSEEKI64)
CHECK_FUNCTION_EXISTS(strerror_r HAVE_STRERROR_R)
CHECK_FUNCTION_EXISTS(strerror_s HAVE_STRERROR_S)
CHECK_FUNCTION_EXISTS(pread HAVE_PREAD)
CHECK_FUNCTION_EXISTS(pwrite HAVE_PWRITE)

add_definitions(-DHAVE_CONFIG_H=1)


#####################################################################
#  set the core C and fortran functions of the library
#####################################################################
set(CALCEPH_C_SOURCE
    calceph.c			calcephspkinterpolhermite.c
    calcephast.c			calcephspkinterpollagrange.c
    calcephbinpckio.c		calcephspkio.c
    calcephchebyshev.c		calcephspkseg1.c
    calcephfileversion.c		calcephspkseg12.c
    calcephinpopfileversion.c	calcephspkseg13.c
    calcephinpopio.c		calcephspkseg17.c
    calcephinpoporientrecord.c	calcephspkseg18.c
    calcephinpoppositionrecord.c	calcephspkseg2.c
    calcephinpoptime.c		calcephspkseg21.c
    calcephio.c			calcephspkseg5.c
    calcephlink.c			calcephstatetype.c
    calcephorientrecord.c		calcephtime.c
    calcephpam.c			calcephtxtfkio.c
    calcephpositionrecord.c		calcephtxtmkio.c
    calcephspice.c			calcephtxtpckio.c
    calcephspicefileversion.c	calcephtxtpckorient.c
    calcephspiceorientrecord.c	calcephversion.c
    calcephspicepositionrecord.c	calcephlocale.c
    f2003calcephtime.c  f2003calcephgetfov.c 
    f2003calcephbinding.c
    f90calcephtime.c f90calcephgetfov.c
    f90calcephbinding.c
    calcephspicetime.c		util.c
    calcephspkseg14.c  calcephidbyname.c
    calcephspkseg19.c   calcephgetfov.c
    calcephtimeparser.c calcephtimelexer.c
    calcephtimeconversionjdcal.c
    calcephtimeconversiontdbtt.c
    calcephtimeconversionutctdb.c calcephtimeconversiondelta.c
    calcephtimeconversionanytdb.c calcephtimeconversiontdbtcb.c
    calcephtimeconversiontdbspacecraft.c
    calcephthread.c
    writephcreateopenclose.c writephdumprestore.c writephspk2seq.c
    writephcomment.c writephutil.c writephspk2par.c
    writephspk3seq.c writephspk3par.c writephspk8seq.c writephspk8par.c
    writephspk9seq.c writephspk9par.c
    writephspk12seq.c writephspk12par.c
    writephspk13seq.c writephspk13par.c
    writephspk14seq.c
    writephspk102seq.c writephspk102par.c
    writephspk103seq.c writephspk103par.c
    writephpck2seq.c writephpck2par.c
    writephpck3seq.c writephpck3par.c
    writephpck102seq.c writephpck102par.c
    writephpck103seq.c writephpck103par.c
    writephcommon.c
    f2003writephbinding.c
)

if (ENABLE_PYTHON)
    CONFIGURE_FILE(setup.py.in setup.py)
endif()


#####################################################################
#  set the directories
#####################################################################
if(UNIX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_LIBDIR "lib")
    set(CMAKE_INSTALL_DATADIR "share")
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_BINDIR "bin")
    set(CMAKE_INSTALL_LIBEXECDIR "libexec")
    set(CMAKE_INSTALL_MANDIR "share/man")
endif()


include_directories("${CMAKE_BINARY_DIR}/src")
add_subdirectory("src")
add_subdirectory("examples")
add_subdirectory("tools")
if (ENABLE_MEX_OCTAVE)
    add_subdirectory("mexapi/packaging/octave")
    add_subdirectory("mexapi/src")
endif()
if (ENABLE_PYTHON)
    add_subdirectory("pythonapi/src")
    add_subdirectory("pythonapi/packaging")
endif()
if (ENABLE_DOC OR ENABLE_PDF)
    add_subdirectory("doc")
endif()
add_subdirectory("doc/man1")

#####################################################################
# enable the tests
#####################################################################
enable_testing()
add_subdirectory("tests")
if (ENABLE_PYTHON  AND ENABLE_PYTHON_TESTS)
    add_subdirectory("pythonapi/tests")
endif()
if (ENABLE_MEX_OCTAVE)
    add_subdirectory("mexapi/tests")
endif()

#####################################################################
# add the packaging
#####################################################################
include(toolchain/packing.cmake)

export(TARGETS calceph FILE "${PROJECT_BINARY_DIR}/calcephTargets.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE calceph)

#####################################################################
# create the developpment package file ...cmake
#####################################################################
include(toolchain/devconfig.cmake)
