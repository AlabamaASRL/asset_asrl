# Preliminaries ################################################################
if(APPLE)
  include(FindPythonEnv)
endif()

set(PYBIND11_CPP_STANDARD -std=c++17)

find_package(Python ${PYVERSION_EXACT} REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python Executable: " ${PYTHON_EXECUTABLE})

# Sub-Components ###############################################################

add_library(bind-pch STATIC pch.cpp)
target_precompile_headers(bind-pch PRIVATE pch.h)
target_link_libraries(bind-pch PUBLIC
  asset-pch
  pybind11_headers
  ${PYTHON_LIBRARIES}
)
target_include_directories(bind-pch PUBLIC
  ${Python_INCLUDE_DIRS}
  ${ASSET_CODE_ROOT}
)

add_subdirectory(Utils)
add_subdirectory(Solvers)
