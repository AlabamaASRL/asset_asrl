# NonLinearProgram #############################################################
add_library(asset-nlp STATIC NonLinearProgram.cpp)
target_link_libraries(asset-nlp PUBLIC asset-pch)
target_precompile_headers(asset-nlp REUSE_FROM asset-pch)

# PSIOPT #######################################################################
add_library(asset-psiopt STATIC PSIOPT.cpp)
target_include_directories(asset-psiopt PUBLIC ${MKL_INCLUDE_DIRS})
target_link_libraries(asset-psiopt PUBLIC
  asset-nlp
  ${MKL_LIBRARIES}
)
target_precompile_headers(asset-psiopt REUSE_FROM asset-pch)

# OptimizationProblem ##########################################################
add_library(asset-optprob STATIC OptimizationProblem.cpp)
target_link_libraries(asset-optprob PUBLIC asset-psiopt)
target_precompile_headers(asset-optprob REUSE_FROM asset-pch)
