

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/OptimalControl)
include_directories(${PROJECT_SOURCE_DIR}/src/Solvers)
include_directories(${PROJECT_SOURCE_DIR}/src/TypeDefs)
include_directories(${PROJECT_SOURCE_DIR}/src/Utils)
include_directories(${PROJECT_SOURCE_DIR}/src/VectorFunctions)
include_directories(${PROJECT_SOURCE_DIR}/src/Integrators)

add_library(asset_extensions STATIC 
ASSET_Extensions.h
ASSET_Extensions.cpp)

add_dependencies(asset_extensions pch)
target_include_directories(asset_extensions PRIVATE ${INCLUDE_DIRS})

message(STATUS  ${INCLUDE_DIRS})


target_compile_options(asset_extensions PUBLIC ${COMPILE_FLAGS})
target_precompile_headers(asset_extensions REUSE_FROM pch)
