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

# Jet ##########################################################################
add_library(bind-jet STATIC BindJet.cpp)
target_include_directories(bind-jet PUBLIC ${MKL_INCLUDE_DIRS})
target_link_libraries(bind-jet PUBLIC
  bind-pch
  ${MKL_LIBRARIES}
)
target_precompile_headers(bind-jet REUSE_FROM bind-pch)

# Solvers ######################################################################
add_library(bind-solvers STATIC BindSolvers.cpp)
target_link_libraries(bind-solvers PUBLIC
  bind-pch
  bind-psiopt
  bind-optprob
  bind-optprobbase
  bind-jet
)
target_precompile_headers(bind-solvers REUSE_FROM bind-pch)
