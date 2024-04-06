add_library(asset-pch STATIC pch.cpp)
target_precompile_headers(asset-pch PRIVATE ${ASSET_INCLUDE_DIR}/ASSET/pch.h)
target_link_libraries(asset-pch PUBLIC asset-headers)

add_subdirectory(Solvers)
add_subdirectory(Utils)
