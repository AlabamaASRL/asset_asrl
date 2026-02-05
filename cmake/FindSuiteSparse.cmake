# FindSuiteSparse.cmake
# Finds SuiteSparse library components
#
# This module defines:
#  SuiteSparse_FOUND - System has SuiteSparse
#  SuiteSparse_INCLUDE_DIRS - SuiteSparse include directories
#  SuiteSparse_LIBRARIES - The SuiteSparse libraries
#  SuiteSparse_LDL_FOUND - LDL component found
#  SuiteSparse_AMD_FOUND - AMD component found
#
# Components that can be requested:
#  LDL - Simple LDL^T factorization for symmetric indefinite matrices
#  AMD - Approximate Minimum Degree ordering
#  CHOLMOD - Supernodal sparse Cholesky
#  UMFPACK - Unsymmetric multifrontal LU
#  KLU - Circuit simulation LU

# If already in cache, be silent
if(SuiteSparse_INCLUDE_DIRS AND SuiteSparse_LIBRARIES)
  set(SuiteSparse_FIND_QUIETLY TRUE)
endif()

# Find SuiteSparse config (required for all components)
find_path(SuiteSparse_CONFIG_INCLUDE_DIR
  NAMES SuiteSparse_config.h
  PATHS
    $ENV{SUITESPARSE_ROOT}/include
    /usr/include/suitesparse
    /usr/local/include/suitesparse
    /usr/include
    /usr/local/include
    "C:/Program Files/SuiteSparse/include"
  PATH_SUFFIXES suitesparse
)

find_library(SuiteSparse_CONFIG_LIBRARY
  NAMES suitesparseconfig
  PATHS
    $ENV{SUITESPARSE_ROOT}/lib
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    "C:/Program Files/SuiteSparse/lib"
  PATH_SUFFIXES suitesparse
)

# Find LDL (primary solver for ASSET)
find_path(SuiteSparse_LDL_INCLUDE_DIR
  NAMES ldl.h
  PATHS
    $ENV{SUITESPARSE_ROOT}/include
    /usr/include/suitesparse
    /usr/local/include/suitesparse
    /usr/include
    /usr/local/include
    "C:/Program Files/SuiteSparse/include"
  PATH_SUFFIXES suitesparse
)

find_library(SuiteSparse_LDL_LIBRARY
  NAMES ldl
  PATHS
    $ENV{SUITESPARSE_ROOT}/lib
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    "C:/Program Files/SuiteSparse/lib"
  PATH_SUFFIXES suitesparse
)

# Find AMD (required by LDL for ordering)
find_path(SuiteSparse_AMD_INCLUDE_DIR
  NAMES amd.h
  PATHS
    $ENV{SUITESPARSE_ROOT}/include
    /usr/include/suitesparse
    /usr/local/include/suitesparse
    /usr/include
    /usr/local/include
    "C:/Program Files/SuiteSparse/include"
  PATH_SUFFIXES suitesparse
)

find_library(SuiteSparse_AMD_LIBRARY
  NAMES amd
  PATHS
    $ENV{SUITESPARSE_ROOT}/lib
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
    "C:/Program Files/SuiteSparse/lib"
  PATH_SUFFIXES suitesparse
)

# Set component found flags
if(SuiteSparse_LDL_INCLUDE_DIR AND SuiteSparse_LDL_LIBRARY)
  set(SuiteSparse_LDL_FOUND TRUE)
endif()

if(SuiteSparse_AMD_INCLUDE_DIR AND SuiteSparse_AMD_LIBRARY)
  set(SuiteSparse_AMD_FOUND TRUE)
endif()

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SuiteSparse
  REQUIRED_VARS
    SuiteSparse_CONFIG_INCLUDE_DIR
    SuiteSparse_CONFIG_LIBRARY
    SuiteSparse_LDL_INCLUDE_DIR
    SuiteSparse_LDL_LIBRARY
    SuiteSparse_AMD_INCLUDE_DIR
    SuiteSparse_AMD_LIBRARY
  HANDLE_COMPONENTS
)

if(SuiteSparse_FOUND)
  # Collect all include directories
  set(SuiteSparse_INCLUDE_DIRS
    ${SuiteSparse_CONFIG_INCLUDE_DIR}
    ${SuiteSparse_LDL_INCLUDE_DIR}
    ${SuiteSparse_AMD_INCLUDE_DIR}
  )
  list(REMOVE_DUPLICATES SuiteSparse_INCLUDE_DIRS)
  
  # Collect all libraries (order matters: LDL depends on AMD and config)
  set(SuiteSparse_LIBRARIES
    ${SuiteSparse_LDL_LIBRARY}
    ${SuiteSparse_AMD_LIBRARY}
    ${SuiteSparse_CONFIG_LIBRARY}
  )
  
  if(NOT SuiteSparse_FIND_QUIETLY)
    message(STATUS "SuiteSparse found:")
    message(STATUS "  Include: ${SuiteSparse_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${SuiteSparse_LIBRARIES}")
    message(STATUS "  LDL: ${SuiteSparse_LDL_FOUND}")
    message(STATUS "  AMD: ${SuiteSparse_AMD_FOUND}")
  endif()
else()
  set(SuiteSparse_INCLUDE_DIRS "")
  set(SuiteSparse_LIBRARIES "")
endif()

mark_as_advanced(
  SuiteSparse_CONFIG_INCLUDE_DIR
  SuiteSparse_CONFIG_LIBRARY
  SuiteSparse_LDL_INCLUDE_DIR
  SuiteSparse_LDL_LIBRARY
  SuiteSparse_AMD_INCLUDE_DIR
  SuiteSparse_AMD_LIBRARY
  SuiteSparse_INCLUDE_DIRS
  SuiteSparse_LIBRARIES
)
