# FindOpenBLAS.cmake
# Finds OpenBLAS library as an alternative to Intel MKL
#
# This module defines:
#  OpenBLAS_FOUND - System has OpenBLAS
#  OpenBLAS_INCLUDE_DIRS - OpenBLAS include directories
#  OpenBLAS_LIBRARIES - The OpenBLAS libraries
#  OpenBLAS_LIBRARY - The main OpenBLAS library
#
# Usage:
#  find_package(OpenBLAS)
#  if(OpenBLAS_FOUND)
#    target_link_libraries(TARGET ${OpenBLAS_LIBRARIES})
#  endif()

# If already in cache, be silent
if(OpenBLAS_INCLUDE_DIRS AND OpenBLAS_LIBRARIES)
  set(OpenBLAS_FIND_QUIETLY TRUE)
endif()

# Find OpenBLAS include directory
find_path(OpenBLAS_INCLUDE_DIR
  NAMES cblas.h openblas_config.h
  PATHS
    $ENV{OPENBLAS_ROOT}/include
    $ENV{OPENBLAS_HOME}/include
    /usr/include
    /usr/local/include
    /opt/OpenBLAS/include
    "C:/Program Files/OpenBLAS/include"
    "C:/OpenBLAS/include"
  PATH_SUFFIXES openblas
)

# Find OpenBLAS library
if(WIN32)
  set(OPENBLAS_LIB_NAMES openblas libopenblas)
else()
  set(OPENBLAS_LIB_NAMES openblas)
endif()

find_library(OpenBLAS_LIBRARY
  NAMES ${OPENBLAS_LIB_NAMES}
  PATHS
    $ENV{OPENBLAS_ROOT}/lib
    $ENV{OPENBLAS_HOME}/lib
    /usr/lib
    /usr/local/lib
    /opt/OpenBLAS/lib
    "C:/Program Files/OpenBLAS/lib"
    "C:/OpenBLAS/lib"
  PATH_SUFFIXES openblas
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS
  DEFAULT_MSG
  OpenBLAS_LIBRARY
  OpenBLAS_INCLUDE_DIR
)

if(OpenBLAS_FOUND)
  set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
  set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})
  
  message(STATUS "OpenBLAS found:")
  message(STATUS "  Include: ${OpenBLAS_INCLUDE_DIRS}")
  message(STATUS "  Library: ${OpenBLAS_LIBRARIES}")
else()
  set(OpenBLAS_LIBRARIES "")
  set(OpenBLAS_INCLUDE_DIRS "")
endif()

mark_as_advanced(
  OpenBLAS_INCLUDE_DIR
  OpenBLAS_LIBRARY
  OpenBLAS_LIBRARIES
  OpenBLAS_INCLUDE_DIRS
)
