# PSIOPT #######################################################################
add_library(bind-psiopt STATIC BindPSIOPT.cpp)
target_link_libraries(bind-psiopt PUBLIC
  bind-pch
  asset-psiopt
  asset-utils
)
target_precompile_headers(bind-psiopt REUSE_FROM bind-pch)
