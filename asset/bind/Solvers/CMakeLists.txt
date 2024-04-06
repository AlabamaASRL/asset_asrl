# PSIOPT #######################################################################
add_library(bind-psiopt STATIC BindPSIOPT.cpp)
target_link_libraries(bind-psiopt PUBLIC
  bind-pch
  asset-psiopt
  asset-utils
)
target_precompile_headers(bind-psiopt REUSE_FROM bind-pch)

# OptimizationProblem ##########################################################
add_library(bind-optprob STATIC BindOptimizationProblem.cpp)
target_link_libraries(bind-optprob PUBLIC
  bind-pch
  asset-optprob
)
target_precompile_headers(bind-optprob REUSE_FROM bind-pch)

# OptimizationProblemBase ######################################################
add_library(bind-optprobbase STATIC BindOptimizationProblemBase.cpp)
target_link_libraries(bind-optprobbase PUBLIC bind-pch)
target_precompile_headers(bind-optprobbase REUSE_FROM bind-pch)
