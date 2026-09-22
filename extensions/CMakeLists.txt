include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/OptimalControl)
include_directories(${PROJECT_SOURCE_DIR}/src/Solvers)
include_directories(${PROJECT_SOURCE_DIR}/src/TypeDefs)
include_directories(${PROJECT_SOURCE_DIR}/src/Utils)
include_directories(${PROJECT_SOURCE_DIR}/src/VectorFunctions)
include_directories(${PROJECT_SOURCE_DIR}/src/Integrators)

# Automatically detect all C++ source and header files in this directory.
# CONFIGURE_DEPENDS makes CMake reconfigure if new files are added.
file(GLOB ASSET_EXTENSION_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)

add_library(asset_extensions STATIC
    ${ASSET_EXTENSION_SOURCES}
)

add_dependencies(asset_extensions pch)

target_include_directories(asset_extensions PRIVATE ${INCLUDE_DIRS})

message(STATUS "ASSET extension sources:")
foreach(SOURCE ${ASSET_EXTENSION_SOURCES})
    message(STATUS "  ${SOURCE}")
endforeach()

target_compile_options(asset_extensions PUBLIC ${COMPILE_FLAGS})
target_precompile_headers(asset_extensions REUSE_FROM pch)
