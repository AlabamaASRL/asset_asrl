# Public-domain NRLMSISE-00 C release 20041227, obtained from NASA CCMC's
# ModelWeb archive. The implementation uses internal mutable work arrays;
# callers must serialize evaluations.
add_library(asset_nrlmsise00 STATIC nrlmsise-00.c nrlmsise-00_data.c)
set_target_properties(asset_nrlmsise00 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(asset_nrlmsise00 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(asset_nrlmsise00 PUBLIC m)
