# Unmodified IAU SOFA ANSI C release 2023-10-11.
# The upstream archive includes its test main in t_sofa_c.c; it is excluded
# from the library. Each source file carries the SOFA license notice.
file(GLOB SOFA_C_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
list(FILTER SOFA_C_SOURCES EXCLUDE REGEX "/t_sofa_c\\.c$")
add_library(asset_sofa STATIC ${SOFA_C_SOURCES})
set_target_properties(asset_sofa PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(asset_sofa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
