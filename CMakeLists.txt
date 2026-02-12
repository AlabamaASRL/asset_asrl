cmake_minimum_required(VERSION 3.16)
project(ASSET VERSION 0.5.2 LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set( CMAKE_MESSAGE_LOG_LEVEL "STATUS" )
################################################################################
################## Includes ####################################################
################################################################################

include(cmake/git-submodule-init.cmake)
include(cmake/git-submodule-update.cmake)


################################################################################
############### Warnings and Errors ############################################
################################################################################

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  message(FATAL_ERROR "You cannot build ASSET with the Microsoft Compiler. Please use Clang or GCC and try again.")
endif()

if(NOT WIN32)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
  endif()
endif()


################################################################################
############### Build Settings #################################################
################################################################################


set(LINK_TIME_OPT FALSE)          ### DO LTO - Recommended for full release
set(CLANG_MAX_INLINE_DEPTH 225)
#set(BUILD_ASSET_WHEEL TRUE)      ### CI Build for generic wheels
#set(BUILD_SPHINX_DOCS TRUE)      


################################################################################
##########################  Set Compiler Flags  ################################
################################################################################

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Release Flags
list(APPEND RELEASE_FLAGS "-O2")


## Generic Binary Flags
if(BUILD_ASSET_WHEEL)
    # Only add x86-64 specific flags if not on ARM64
    if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64"))
        #list(APPEND RELEASE_FLAGS "-mcx16;-mpopcnt;-msse3;-msse4.1;-msse4.2;-mssse3;-mavx;-mavx2;-mbmi;-mbmi2;-mf16c;-mfma;-mlzcnt;-mmovbe;-mxsave")
        # x86-64-v3 flags written out for older compilers
        list(APPEND RELEASE_FLAGS "-mcx16")
        list(APPEND RELEASE_FLAGS "-mpopcnt")
        list(APPEND RELEASE_FLAGS "-msse3")
        list(APPEND RELEASE_FLAGS "-msse4.1")
        list(APPEND RELEASE_FLAGS "-msse4.2")
        list(APPEND RELEASE_FLAGS "-mssse3")
        list(APPEND RELEASE_FLAGS "-mavx")
        list(APPEND RELEASE_FLAGS "-mavx2")
        list(APPEND RELEASE_FLAGS "-mbmi")
        list(APPEND RELEASE_FLAGS "-mbmi2")
        list(APPEND RELEASE_FLAGS "-mf16c")
        list(APPEND RELEASE_FLAGS "-mfma")
        list(APPEND RELEASE_FLAGS "-mlzcnt")
        list(APPEND RELEASE_FLAGS "-mmovbe")
        list(APPEND RELEASE_FLAGS "-mxsave")
    endif()

    # Windows runs out of ram with LINK_TIME_OPT on gh-actions
    if(NOT WIN32)
        list(APPEND RELEASE_FLAGS "-mtune=skylake")
    endif()
    set(LINK_TIME_OPT TRUE)   ### DO LTO - Recommended for full release
    set(CLANG_MAX_INLINE_DEPTH 400)
    
else()
    # -march=native and -mtune=native work on Linux/macOS ARM64 (GCC/Clang)
    # but not on Windows ARM64 with clang-cl (which uses MSVC-style flags)
    if(NOT (WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64"))
        list(APPEND RELEASE_FLAGS "-march=native")
        if(NOT WIN32)
            list(APPEND RELEASE_FLAGS "-mtune=native")
        endif()
    endif()
endif()


if(WIN32)
     list(APPEND RELEASE_FLAGS "/MD")
     list(APPEND RELEASE_FLAGS "/DNDEBUG")
     list(APPEND RELEASE_FLAGS "/O2")
     list(APPEND RELEASE_FLAGS "/Ob2")
     # Only use AVX2 on x86/x64, not on ARM64
     if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64"))
         list(APPEND RELEASE_FLAGS "/arch:AVX2")
     endif()
     list(APPEND RELEASE_FLAGS "/fp:fast")
     list(APPEND RELEASE_FLAGS "/GS-")
     list(APPEND RELEASE_FLAGS "/Gw")
endif()

if(NOT WIN32)
    # Debugging Flags
    set(DEBUG_FLAGS "-g" "-ggdb3")
    # Common Flags
    list(APPEND COMMON_FLAGS "-pthread")
endif()
 

if(LINK_TIME_OPT)
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if(WIN32)
        ## clang-cl flag
        if(BUILD_ASSET_WHEEL)
            ## Do full LTO for CI builds since it is single threaded anyway
            list(APPEND RELEASE_FLAGS "-flto=full")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=1")
        else()
            list(APPEND RELEASE_FLAGS "-flto=thin")
            list(APPEND RELEASE_FLAGS "/opt:lldltojobs=8")
        endif()
    else()
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    set(CMAKE_CXX_COMPILE_OPTIONS_IPO ${CMAKE_CXX_COMPILE_OPTIONS_IPO} -flto=1)
  endif()

else()
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
endif()



list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-mllvm>")
list(APPEND COMMON_FLAGS "$<$<STREQUAL:${CMAKE_CXX_COMPILER_ID},Clang>:-inline-threshold=${CLANG_MAX_INLINE_DEPTH}>")



if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND FALSE)
    list(APPEND RELEASE_FLAGS "-finline-limit=1000")
    list(APPEND RELEASE_FLAGS "--param=inline-unit-growth=500")    #40,40
    list(APPEND RELEASE_FLAGS "--param=early-inlining-insns=50")   #14,14
    list(APPEND RELEASE_FLAGS "--param=large-function-insns=3500") #2700,2700
    list(APPEND RELEASE_FLAGS "--param=large-function-growth=120") #100,100
    list(APPEND RELEASE_FLAGS "--param=inline-min-speedup=10")     #30,15
endif()


if(NOT WIN32)
  list(APPEND RELEASE_FLAGS "-fomit-frame-pointer")
  list(APPEND RELEASE_FLAGS "-fno-stack-protector")
  #list(APPEND RELEASE_FLAGS "-fno-stack-clash-protection")
  #list(APPEND RELEASE_FLAGS "-fcf-protection=none")
  list(APPEND RELEASE_FLAGS "-fno-asynchronous-unwind-tables")
  list(APPEND RELEASE_FLAGS "-ffast-math")
endif()
 
if(APPLE)
  list(APPEND COMMON_FLAGS "-Xlinker -undefined")
  list(APPEND COMMON_FLAGS "-Xlinker dynamic_lookup")
endif()



# Combine Flags
set(COMPILE_FLAGS PUBLIC ${COMMON_FLAGS})
list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>:${RELEASE_FLAGS}>")
list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>:${DEBUG_FLAGS}>")



################################################################################
################## ASSET PREPROCCESSOR MACROS  #################################
################################################################################

add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO) ##NEVER EVER GET RID OF THIS EVER
add_definitions(-DEIGEN_DONT_PARALLELIZE) ##NEVER EVER GET RID OF THIS EVER


add_definitions(-DASSET_DEFAULT_QP_THREADS=8) ## no spaces between = !!!! otherwise it is set to one
add_definitions(-DASSET_DEFAULT_FUNC_THREADS=16) ## no spaces between = !!!! otherwise it is set to one

###  initial size of asset arena memory allocator in number of doubles/superscalars
add_definitions(-DASSET_DEFAULT_ARENA_SIZE=256) ## no spaces between = !!!! otherwise it is set to one

### Comment out to roll back to old memory allocator. We will get rid of this once confident in arena
add_definitions(-DASSET_MEMORYMAN)

add_definitions(-DASSET_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DASSET_VERSIONSTRING="${CMAKE_PROJECT_VERSION}")
add_definitions(-DASSET_COMPILERSTRING="${CMAKE_CXX_COMPILER_ID}")
add_definitions(-DASSET_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}")
add_definitions(-DASSET_OS="${CMAKE_SYSTEM_NAME}")
add_definitions(-DASSET_OSVERSION="${CMAKE_SYSTEM_VERSION}")
add_definitions(-DFMT_HEADER_ONLY)

if(WIN32)
## See : https://developercommunity.visualstudio.com/t/Access-violation-in-_Thrd_yield-after-up/10664660#T-N10669129-N10678728
add_definitions(-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()



################################################################################
###################### Find and set our dependencies ###########################
################################################################################

if(APPLE)
  include(FindPythonEnv)
endif()
# Set up submodule dependencies
include_directories(dep/eigen)
include_directories(dep/fmt/include)
include_directories(dep/autodiff)

add_subdirectory(dep)

set(PYBIND11_CPP_STANDARD -std=c++17)

# Set up external dependencies
find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
  # find_package(OpenMP REQUIRED)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND COMPILE_FLAGS "-fopenmp=libiomp5")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GCC")
    list(APPEND COMPILE_FLAGS "-fopenmp")
  endif()
elseif(APPLE)
  if(FALSE)
    include_directories("/usr/local/opt/llvm/include")
    link_directories("/usr/local/opt/llvm/lib")
    set(CMAKE_LIBRARY_PATH /usr/local/opt/llvm/lib ${CMAKE_LIBRARY_PATH})
    set(CMAKE_LIBRARY_PATH $ENV{MKLROOT}/lib ${CMAKE_LIBRARY_PATH})
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(OpenMP_C "${CMAKE_C_COMPILER}")
      set(OpenMP_C_FLAGS "-fopenmp=libomp")
      set(OpenMP_C_LIB_NAMES "libomp")
      set(OpenMP_libomp_LIBRARY "omp")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
      set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
      set(OpenMP_CXX_LIB_NAMES "libomp")
      set(OpenMP_libomp_LIBRARY "omp")
      find_package(OpenMP REQUIRED)
      list(APPEND COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
    endif()

    
  endif()
  
endif()

# Detect ARM64 platforms where MKL is not available
# MKL supports: Linux x86_64, macOS x86_64/ARM64, Windows x86/x64
# MKL does NOT support: Windows ARM64, Linux ARM64 (aarch64)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64|arm64")
  if(NOT APPLE)
    # Windows ARM64 and Linux ARM64 lack MKL support
    set(USE_MKL_ALTERNATIVE TRUE)
    message(STATUS "Detected ${CMAKE_SYSTEM_NAME} ARM64 - Using OpenBLAS + SuiteSparse instead of MKL")
  else()
    # macOS ARM64 (Apple Silicon) has MKL support via Rosetta/universal builds
    set(USE_MKL_ALTERNATIVE FALSE)
  endif()
else()
  set(USE_MKL_ALTERNATIVE FALSE)
endif()

# Allow user override: cmake -DFORCE_OPENBLAS=ON to skip MKL even on x86
if(FORCE_OPENBLAS)
  set(USE_MKL_ALTERNATIVE TRUE)
  message(STATUS "FORCE_OPENBLAS=ON - Using OpenBLAS + SuiteSparse instead of MKL")
endif()

# Find appropriate BLAS library and sparse solver
if(NOT USE_MKL_ALTERNATIVE)
  # Use Intel MKL on supported platforms
  find_package(MKL REQUIRED)
  set(ASSET_HAS_MKL TRUE)
  set(ASSET_HAS_OPENBLAS FALSE)
  set(ASSET_HAS_SUITESPARSE FALSE)
  add_definitions(-DASSET_HAS_MKL)
  message(STATUS "Using Intel MKL for BLAS/LAPACK and PARDISO")
else()
  # Use OpenBLAS + SuiteSparse on platforms without MKL
  find_package(OpenBLAS REQUIRED)
  find_package(SuiteSparse REQUIRED)
  set(ASSET_HAS_MKL FALSE)
  set(ASSET_HAS_OPENBLAS TRUE)
  set(ASSET_HAS_SUITESPARSE TRUE)
  add_definitions(-DASSET_HAS_OPENBLAS)
  add_definitions(-DASSET_HAS_SUITESPARSE)
  message(STATUS "Using OpenBLAS for BLAS/LAPACK")
  message(STATUS "Using SuiteSparse LDL for sparse solver (PARDISO replacement)")
endif()

# Find Python with flexible version detection
# PYVERSION_EXACT can be set to lock to a specific version (e.g., -DPYVERSION_EXACT=3.11)
# If not set, CMake will find any compatible Python 3.x version
if(PYVERSION_EXACT)
  message(STATUS "Looking for Python version: ${PYVERSION_EXACT}")
  find_package(Python ${PYVERSION_EXACT} EXACT REQUIRED COMPONENTS Interpreter Development)
else()
  # Find any Python 3.x version with development headers
  find_package(Python 3 REQUIRED COMPONENTS Interpreter Development)
endif()

# Verify Python was found with development headers
if(NOT Python_FOUND)
  message(FATAL_ERROR "Python not found. Please install Python with development headers.")
endif()

if(NOT Python_Development_FOUND)
  message(FATAL_ERROR 
    "Python development headers (Python.h) not found.\n"
    "Solutions:\n"
    "  1. Install Python development package:\n"
    "     - Windows: Reinstall Python with 'Include development headers' option\n"
    "     - Linux: Install python3-dev or python3-devel\n"
    "     - macOS: Python from python.org includes headers by default\n"
    "  2. Set Python path explicitly: cmake -DPython_ROOT_DIR=/path/to/python\n"
    "  3. Set specific version: cmake -DPYVERSION_EXACT=3.11"
  )
endif()

message(STATUS "Python found: ${Python_VERSION} at ${Python_EXECUTABLE}")
message(STATUS "Python include: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python_LIBRARIES}")

# Verify Python architecture matches target architecture on ARM64
if(USE_MKL_ALTERNATIVE AND WIN32)
  # Check if Python library path contains indication of architecture
  string(FIND "${Python_LIBRARIES}" "arm64" ARM64_FOUND)
  string(FIND "${Python_LIBRARIES}" "ARM64" ARM64_FOUND_UPPER)
  
  if(ARM64_FOUND EQUAL -1 AND ARM64_FOUND_UPPER EQUAL -1)
    message(WARNING 
      "Building for Windows ARM64 but Python libraries may not be ARM64.\n"
      "Python library: ${Python_LIBRARIES}\n"
      "\n"
      "If you get linker errors about architecture mismatch (arm64 vs x64):\n"
      "  1. Install ARM64 Python from: https://www.python.org/downloads/windows/\n"
      "     (Look for 'Windows installer (ARM64)' download)\n"
      "  2. Set Python path to ARM64 installation:\n"
      "     cmake -DPython_ROOT_DIR=\"C:\\Python311-ARM64\"\n"
      "  3. Verify with: python -c \"import platform; print(platform.machine())\"\n"
      "     (Should output 'ARM64')"
    )
  else()
    message(STATUS "Python appears to be ARM64 compatible")
  endif()
endif()

# Set dependency variables
# Note: Using Python_INCLUDE_DIRS and Python_LIBRARIES (modern CMake FindPython module variables)
if(ASSET_HAS_MKL)
  set(INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR} ${Python_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS})
  set(LINK_LIBS ${Python_LIBRARIES} ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
elseif(ASSET_HAS_OPENBLAS AND ASSET_HAS_SUITESPARSE)
  set(INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR} ${Python_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIRS} ${SuiteSparse_INCLUDE_DIRS})
  set(LINK_LIBS ${Python_LIBRARIES} ${OpenBLAS_LIBRARIES} ${SuiteSparse_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
elseif(ASSET_HAS_OPENBLAS)
  set(INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR} ${Python_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIRS})
  set(LINK_LIBS ${Python_LIBRARIES} ${OpenBLAS_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
else()
  # Fallback (shouldn't reach here)
  set(INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR} ${Python_INCLUDE_DIRS})
  set(LINK_LIBS ${Python_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(UNIX)
  list(APPEND LINK_LIBS m)
endif()



################################################################################
################ Linker Flags ##################################################
if(ASSET_HAS_MKL)
  if(BUILD_SHARED_LIBS AND NOT APPLE AND NOT WIN32)  # Windows and Linux Dynamic Linking
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl, --end-group")
  elseif(UNIX AND NOT APPLE)  # Linux Static Linking
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl,--end-group")
  elseif(APPLE)  # Apple Dynamic and Static Linking
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Xlinker -undefined -Xlinker dynamic_lookup")
  else()  # Windows Static Linking with MKL
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
else()
  # No MKL available (e.g., Windows ARM64)
  if(APPLE)
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Xlinker -undefined -Xlinker dynamic_lookup")
  else()
    list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
endif()

###########################################################
######## Find Correct Python Install Location  ############


if(NOT PIP_INSTALL)
  if(WIN32)
    set(PYTHON_LOCAL_INSTALL_DIR ${Python_SITEARCH})
    
  else()
    execute_process ( COMMAND ${Python_EXECUTABLE} -m site --user-site
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(PYTHON_LOCAL_INSTALL_DIR ${PYTHON_SITE_PACKAGES})
  endif()

  if(PYTHON_LOCAL_INSTALL_DIR)
    message(STATUS "ASSET will be installed to " ${PYTHON_LOCAL_INSTALL_DIR})
  else()
    message(STATUS "Could not find appropriate install directory")
  endif()
endif()

message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS ${OpenMP_CXX_FLAGS})
message(STATUS ${OpenMP_CXX_LIBRARIES})
message(STATUS "Python executable: ${Python_EXECUTABLE}")

################################################################################
# Code Directories
add_subdirectory(src)

add_subdirectory(extensions)


# Documentation
# READ: pip install sphinx,breathe,furo before trying to build docs

if(BUILD_SPHINX_DOCS)
  add_subdirectory(doc)
endif()

add_subdirectory(pypiwheel)

# Formatting
include(cmake/clang-format.cmake)

