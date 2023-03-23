cmake_minimum_required(VERSION 3.16)
project(ASSET VERSION 0.2.0 LANGUAGES CXX)
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
    list(APPEND RELEASE_FLAGS "-march=x86-64-v3")
    # Windows runs out of ram with LINK_TIME_OPT on gh-actions
    if(NOT WIN32)
        list(APPEND RELEASE_FLAGS "-mtune=skylake")
        set(LINK_TIME_OPT TRUE)   ### DO LTO - Recommended for full release
        set(CLANG_MAX_INLINE_DEPTH 300)
    endif()
    
else()
    list(APPEND RELEASE_FLAGS "-march=native")
    if(NOT WIN32)
        list(APPEND RELEASE_FLAGS "-mtune=native")
    endif()
endif()


if(WIN32)
     list(APPEND RELEASE_FLAGS "/MD")
     list(APPEND RELEASE_FLAGS "/DNDEBUG")
     list(APPEND RELEASE_FLAGS "/O2")
     list(APPEND RELEASE_FLAGS "/Ob2")
     list(APPEND RELEASE_FLAGS "/arch:AVX2")
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
  list(APPEND RELEASE_FLAGS "-fno-stack-clash-protection")
  list(APPEND RELEASE_FLAGS "-fcf-protection=none")
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

################################################################################
###################### Find and set our dependencies ###########################
################################################################################

if(APPLE)
  include(FindPythonEnv)
endif()
# Set up submodule dependencies
include_directories(dep/eigen)
include_directories(dep/fmt/include)

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

find_package(MKL REQUIRED)

find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)

# Set dependency variables
set(INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS})
set(LINK_LIBS ${PYTHON_LIBRARIES} ${MKL_LIBRARIES} Threads::Threads ${CMAKE_DL_LIBS})
if(UNIX)
  list(APPEND LINK_LIBS m)
endif()



################################################################################
################ Linker Flags ##################################################
if(BUILD_SHARED_LIBS AND NOT APPLE AND NOT WIN32)  # Windows and Linux Dynamic Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl, --end-group")
elseif(UNIX AND NOT APPLE)  # Linux Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Wl,--no-undefined -Wl,--start-group ${MKL_LIBRARIES_LIST} -Wl,--end-group")
elseif(APPLE)  # Apple Dynamic and Static Linking

  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS} -Xlinker -undefined -Xlinker dynamic_lookup")
else()  # Windows Static Linking
  list(APPEND LINKER_FLAGS "${OpenMP_CXX_FLAGS}")
endif()

###########################################################
######## Find Correct Python Install Location  ############


if(NOT PIP_INSTALL)
  if(WIN32)
    set(PYTHON_LOCAL_INSTALL_DIR ${Python_SITEARCH})
    
  else()
    execute_process ( COMMAND ${PYTHON_EXECUTABLE} -m site --user-site
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
message(STATUS ${PYTHON_EXECUTABLE})

################################################################################
# Code Directories
add_subdirectory(src)

# Documentation
# READ: pip install sphinx,breathe,furo before trying to build docs

if(BUILD_SPHINX_DOCS)
  add_subdirectory(doc)
endif()

add_subdirectory(pypiwheel)

# Formatting
include(cmake/clang-format.cmake)

