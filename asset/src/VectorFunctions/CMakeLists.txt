add_library(asset-dynstack STATIC DynamicStack.cpp)
target_link_libraries(asset-dynstack PUBLIC asset-pch)
target_precompile_headers(asset-dynstack REUSE_FROM asset-pch)

add_library(asset-dynsum STATIC DynamicSum.cpp)
target_link_libraries(asset-dynsum PUBLIC asset-pch)
target_precompile_headers(asset-dynsum REUSE_FROM asset-pch)
