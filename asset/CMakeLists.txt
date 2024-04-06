# Common #######################################################################
set(ASSET_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(ASSET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Header Interface #############################################################
add_library(asset-headers INTERFACE)
target_include_directories(asset-headers INTERFACE
  ${ASSET_INCLUDE_DIR}
  ${EIGEN_INCLUDE_DIR}
)
target_compile_definitions(asset-headers INTERFACE
  # Eigen
  EIGEN_INITIALIZE_MATRICES_BY_ZERO
  EIGEN_DONT_PARALLELIZE
  # ASSET
  ASSET_DEFAULT_QP_THREADS=8
  ASSET_DEFAULT_FUNC_THREADS=16
  ASSET_DEFAULT_ARENA_SIZE=256
  ASSET_MEMORYMAN
  ASSET_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}"
  ASSET_VERSIONSTRING="${CMAKE_PROJECT_VERSION}"
  ASSET_COMPILERSTRING="${CMAKE_CXX_COMPILER_ID}"
  ASSET_COMPILERVERSION="${CMAKE_CXX_COMPILER_VERSION}"
  ASSET_OS="${CMAKE_SYSTEM_NAME}"
  ASSET_OSVERSION="${CMAKE_SYSTEM_VERSION}"
)
target_link_libraries(asset-headers INTERFACE
  fmt::fmt-header-only
  ${MKL_LIBRARIES}
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
target_compile_options(asset-headers INTERFACE ${COMPILE_FLAGS})

if(UNIX)
  target_link_libraries(asset-headers INTERFACE m)
endif()

# Library ######################################################################
add_subdirectory(src)

# Bindings #####################################################################
if(ASSET_PYTHON)
  add_subdirectory(bind)
endif()
