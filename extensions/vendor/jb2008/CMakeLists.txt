enable_language(Fortran)

add_library(asset_jb2008 STATIC JB2008.f jb2008_c_api.f90)
set_target_properties(asset_jb2008 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(asset_jb2008 PRIVATE
    -ffixed-line-length-none -ffree-line-length-none -frecursive)
endif()
