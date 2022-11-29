################################################################################
#
# \file      FindLMKL.cmake
# \copyright 2012-2015 J. Bakosi,
#            2016-2018 Los Alamos National Security, LLC.,
#            2019-2020 Triad National Security, LLC.
#            All rights reserved. See the LICENSE file for details.
# \brief     Find the Math Kernel Library from Intel
#
################################################################################

# Find the Math Kernel Library from Intel
#
#  MKL_FOUND - System has MKL
#  MKL_INCLUDE_DIRS - MKL include files directories
#  MKL_LIBRARIES - The MKL libraries
#  MKL_INTERFACE_LIBRARY - MKL interface library
#  MKL_SEQUENTIAL_LAYER_LIBRARY - MKL sequential layer library
#  MKL_CORE_LIBRARY - MKL core library
#
#  The environment variables MKLROOT and INTEL are used to find the library.
#  Everything else is ignored. If MKL is found "-DMKL_ILP64" is added to
#  CMAKE_C_FLAGS and CMAKE_CXX_FLAGS.
#
#  Example usage:
#
#  find_package(MKL)
#  if(MKL_FOUND)
#    target_link_libraries(TARGET ${MKL_LIBRARIES})
#  endif()

# If already in cache, be silent
if (MKL_INCLUDE_DIRS AND MKL_LIBRARIES AND MKL_INTERFACE_LIBRARY AND
    MKL_SEQTHR_LIBRARY AND MKL_CORE_LIBRARY)
  #set (MKL_FIND_QUIETLY TRUE)
endif()

if(UNIX AND NOT APPLE)  # If Linux
  if(BUILD_SHARED_LIBS)
    set(INT_LIB "libmkl_intel_lp64.so")
    set(SEQ_LIB "libmkl_sequential.so")
    set(COR_LIB "libmkl_core.so")
    set(OMP_LIB "libiomp5.so")
    if("gomp" IN_LIST OpenMP_CXX_LIB_NAMES)
      set(THR_LIB "libmkl_gnu_thread.so")
    else() # Default
      set(THR_LIB "libmkl_intel_thread.so")
    endif()
  else()
    set(INT_LIB "libmkl_intel_lp64.a")
    set(SEQ_LIB "libmkl_sequential.a")
    set(COR_LIB "libmkl_core.a")
    set(OMP_LIB "libiomp5.so")
    if("gomp" IN_LIST OpenMP_CXX_LIB_NAMES)
      set(THR_LIB "libmkl_gnu_thread.a")
    else() # Default
      set(THR_LIB "libmkl_intel_thread.a")
    endif()
  endif()
elseif(APPLE)
  if(BUILD_SHARED_LIBS)
    set(INT_LIB "libmkl_intel_lp64.dylib")
    set(SEQ_LIB "libmkl_sequential.dylib")
    set(COR_LIB "libmkl_core.dylib")
    set(THR_LIB "libmkl_intel_thread.dylib")
    set(OMP_LIB "libiomp5.dylib")

  else()
    set(INT_LIB "libmkl_intel_lp64.a")
    set(SEQ_LIB "libmkl_sequential.a")
    set(COR_LIB "libmkl_core.a")
    set(THR_LIB "libmkl_intel_thread.a")
    set(OMP_LIB "libiomp5.dylib")

  endif()
else()  # if Windows
  if(BUILD_SHARED_LIBS)
    set(INT_LIB "mkl_intel_lp64_dll.lib")
    set(SEQ_LIB "mkl_sequential_dll.lib")
    set(THR_LIB "mkl_intel_thread_dll.lib")
    set(TBBTHR_LIB "mkl_tbb_thread_dll.lib")
    set(COR_LIB "mkl_core_dll.lib")
    set(OMP_LIB "libiomp5md.lib")
    set(TBB_LIB "tbb12.lib")
  else()
    set(INT_LIB "mkl_intel_lp64.lib")
    set(SEQ_LIB "mkl_sequential.lib")
    set(THR_LIB "mkl_intel_thread.lib")
    set(TBBTHR_LIB "mkl_tbb_thread.lib")
    set(COR_LIB "mkl_core.lib")
    set(OMP_LIB "libiomp5md.lib")
    set(TBB_LIB "tbb12.lib")
  endif()
endif() # End OS Conditional

find_path(MKL_INCLUDE_DIR NAMES mkl.h
                          HINTS $ENV{MKLROOT}/include /opt/intel/mkl/include   $ENV{ONEAPI_ROOT}/mkl/latest/include

                          PATH_SUFFIXES mkl)

message(STATUS ${OMP_LIB})
if(WIN32 AND MKL_USE_TBB)

  find_path(TBB_INCLUDE_DIR NAMES tbb.h
                            HINTS $ENV{TBBROOT}/include
                            PATH_SUFFIXES tbb)

  set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR} $ENV{TBBROOT}/include )

else()
  set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
endif()


find_library(MKL_INTERFACE_LIBRARY
            NAMES ${INT_LIB}
            PATHS $ENV{MKLROOT}/lib
                  $ENV{MKLROOT}/lib/intel64
                  $ENV{INTEL}/mkl/lib/intel64
                  $ENV{INTEL}/mkl/latest/lib/intel64
                  $ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64
                  $ENV{MKLROOT}/lib/intel64_win
                  $ENV{INTEL}/mkl/lib/intel64_win)

find_library(MKL_CORE_LIBRARY
            NAMES ${COR_LIB}
            PATHS $ENV{MKLROOT}/lib
                  $ENV{MKLROOT}/lib/intel64
                  $ENV{INTEL}/mkl/lib/intel64
                  $ENV{INTEL}/mkl/latest/lib/intel64
                  $ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64
                  $ENV{MKLROOT}/lib/intel64_win
                  $ENV{INTEL}/mkl/lib/intel64_win)

find_library(MKL_OMP_LIBRARY
            NAMES ${OMP_LIB}
            PATHS $ENV{MKLROOT}/lib
                  $ENV{MKLROOT}/lib/intel64
                  $ENV{INTEL}/mkl/lib/intel64
                  $ENV{INTEL}/mkl/latest/lib/intel64
                  $ENV{INTEL}/compiler/lib/intel64
                  $ENV{INTEL}/compiler/lib/intel64_win
                  $ENV{INTEL}/compiler/latest/lib/intel64
                  $ENV{INTEL}/compiler/latest/lib/intel64_win
                  $ENV{ONEAPI_ROOT}/compiler/latest/linux/compiler/lib/intel64_lin
                  $ENV{ONEAPI_ROOT}/compiler/latest/windows/compiler/lib/intel64_win
                  $ENV{ONEAPI_ROOT}/compiler/2022.0.0/mac/compiler/lib
                  /opt/intel/oneapi/compiler/latest/mac/compiler/lib
                  $ENV{INTEL}/compiler/latest/linux/compiler/lib/intel64_lin)  # atrocious



if(MKL_USE_SEQUENTIAL)
  find_library(MKL_SEQTHR_LIBRARY
  NAMES ${SEQ_LIB}
  PATHS $ENV{MKLROOT}/lib
        $ENV{MKLROOT}/lib/intel64
        $ENV{INTEL}/mkl/lib/intel64
        $ENV{MKLROOT}/lib/intel64_win
        $ENV{INTEL}/mkl/lib/intel64_win)
elseif(WIN32 AND MKL_USE_TBB)
  find_library(MKL_SEQTHR_LIBRARY
  NAMES ${TBBTHR_LIB}
  PATHS $ENV{MKLROOT}/lib
        $ENV{MKLROOT}/lib/intel64
        $ENV{INTEL}/mkl/lib/intel64
        $ENV{MKLROOT}/lib/intel64_win
        $ENV{INTEL}/mkl/lib/intel64_win)

  find_library(MKL_TBB_LIBRARY
  NAMES ${TBB_LIB}
  PATHS $ENV{TBBROOT}/lib
        $ENV{TBBROOT}/lib/intel64/vc_mt
        $ENV{INTEL}/mkl/lib/intel64)
else()
  find_library(MKL_SEQTHR_LIBRARY
  NAMES ${THR_LIB}
  PATHS $ENV{MKLROOT}/lib
        $ENV{MKLROOT}/lib/intel64
        $ENV{INTEL}/mkl/lib/intel64
        $ENV{MKLROOT}/lib/intel64_win
        $ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64
        $ENV{INTEL}/mkl/lib/intel64_win)
endif()

message(STATUS "MKL_INTERFACE_LIBRARY = ${MKL_INTERFACE_LIBRARY}")
message(STATUS "MKL_SEQTHR_LIBRARY = ${MKL_SEQTHR_LIBRARY}")
message(STATUS "MKL_CORE_LIBRARY = ${MKL_CORE_LIBRARY}")
message(STATUS "MKL_TBB_LIBRARY = ${MKL_TBB_LIBRARY}")
message(STATUS "MKL_OMP_LIBRARY = ${MKL_OMP_LIBRARY}")

if(UNIX)
  set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQTHR_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_OMP_LIBRARY})
  list(APPEND MKL_LIBRARIES_LIST ${MKL_INTERFACE_LIBRARY} " " ${MKL_SEQTHR_LIBRARY} " " ${MKL_CORE_LIBRARY} " " ${MKL_OMP_LIBRARY})
elseif(WIN32 AND MKL_USE_TBB)
  set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQTHR_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_TBB_LIBRARY})
  list(APPEND MKL_LIBRARIES_LIST ${MKL_INTERFACE_LIBRARY} " " ${MKL_SEQTHR_LIBRARY} " " ${MKL_CORE_LIBRARY}" " ${MKL_TBB_LIBRARY})
else()
  set(MKL_LIBRARIES ${MKL_INTERFACE_LIBRARY} ${MKL_SEQTHR_LIBRARY} ${MKL_CORE_LIBRARY} ${MKL_OMP_LIBRARY})
  list(APPEND MKL_LIBRARIES_LIST ${MKL_INTERFACE_LIBRARY} " " ${MKL_SEQTHR_LIBRARY} " " ${MKL_CORE_LIBRARY} " " ${MKL_OMP_LIBRARY})
endif()

if (MKL_INCLUDE_DIR AND
    MKL_INTERFACE_LIBRARY AND
    MKL_SEQTHR_LIBRARY AND
    MKL_CORE_LIBRARY)

    if (NOT DEFINED ENV{CRAY_PRGENVPGI} AND
        NOT DEFINED ENV{CRAY_PRGENVGNU} AND
        NOT DEFINED ENV{CRAY_PRGENVCRAY} AND
        NOT DEFINED ENV{CRAY_PRGENVINTEL})
      set(ABI "-m64")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMKL_LP64 ${ABI}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMKL_LP64 ${ABI}")

else()

  set(MKL_INCLUDE_DIRS "")
  set(MKL_LIBRARIES "")
  set(MKL_INTERFACE_LIBRARY "")
  set(MKL_SEQTHR_LIBRARY "")
  set(MKL_CORE_LIBRARY "")

endif()

# Add MKL library path to rpath
if(MKL_CORE_LIBRARY)
  get_filename_component(MKL_RPATH_DIR ${MKL_CORE_LIBRARY} DIRECTORY)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${MKL_RPATH_DIR}")
endif()

# Handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIRS MKL_INTERFACE_LIBRARY MKL_SEQTHR_LIBRARY MKL_CORE_LIBRARY)

MARK_AS_ADVANCED(MKL_INCLUDE_DIRS MKL_LIBRARIES MKL_LIBRARIES_LIST MKL_INTERFACE_LIBRARY MKL_SEQTHR_LIBRARY MKL_CORE_LIBRARY)
