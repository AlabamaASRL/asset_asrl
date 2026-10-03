

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/OptimalControl)
include_directories(${PROJECT_SOURCE_DIR}/src/Solvers)
include_directories(${PROJECT_SOURCE_DIR}/src/TypeDefs)
include_directories(${PROJECT_SOURCE_DIR}/src/Utils)
include_directories(${PROJECT_SOURCE_DIR}/src/VectorFunctions)
include_directories(${PROJECT_SOURCE_DIR}/src/Integrators)
add_subdirectory(vendor/sofa)
add_subdirectory(vendor/nrlmsise00)
add_subdirectory(vendor/jb2008)

# CALCEPH 5.0.1 is vendored unchanged and built as a private static dependency.
# Its optional language bindings, installation rules, and package registry are
# unnecessary for the ASSET extension module.
set(ENABLE_FORTRAN OFF)
set(ENABLE_PYTHON OFF)
set(ENABLE_MEX_OCTAVE OFF)
set(CALCEPH_INSTALL OFF)
set(ENABLE_DOC OFF)
set(ENABLE_PDF OFF)
set(ENABLE_THREAD ON)
set(CALCEPH_USE_PIC ON)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
set(_asset_build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(vendor/calceph EXCLUDE_FROM_ALL)
target_include_directories(calceph PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/vendor/calceph/src)
set(BUILD_SHARED_LIBS ${_asset_build_shared_libs})

add_library(asset_extensions STATIC 
ASSET_Extensions.h
ASSET_Extensions.cpp
PinesGravity.cpp
PinesGravityModel.cpp
PinesGravityModel.h
PanelDrag.cpp
GeodeticWGS84.cpp
EarthFrame.cpp)
target_sources(asset_extensions PRIVATE
NRLMSISE00Density.cpp
DE430Ephemeris.cpp
DE430Position.cpp
ThirdBodyGravity.cpp)
target_sources(asset_extensions PRIVATE CannonballSRP.cpp)
target_sources(asset_extensions PRIVATE JB2008Density.cpp SmoothAtmosphereBlend.cpp)

add_dependencies(asset_extensions pch)
target_include_directories(asset_extensions PRIVATE ${INCLUDE_DIRS})

message(STATUS  ${INCLUDE_DIRS})


target_compile_options(asset_extensions PUBLIC ${COMPILE_FLAGS})
target_precompile_headers(asset_extensions REUSE_FROM pch)
target_link_libraries(asset_extensions PUBLIC asset_sofa asset_nrlmsise00 asset_jb2008 calceph)
