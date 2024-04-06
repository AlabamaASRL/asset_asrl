add_library(asset-utils STATIC
  fmtlib.cpp
  ColorText.cpp
  GetCoreCount.cpp
  MemoryManagement.cpp
)
target_link_libraries(asset-utils PUBLIC asset-headers)
target_precompile_headers(asset-utils REUSE_FROM asset-pch)
