add_library(bind-utils STATIC BindMemoryManagement.cpp)
target_link_libraries(bind-utils PUBLIC
  bind-pch
  asset-utils
)
target_precompile_headers(bind-utils REUSE_FROM bind-pch)
