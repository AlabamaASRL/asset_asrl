# NonLinearProgram #############################################################
add_library(asset-nlp STATIC NonLinearProgram.cpp)
target_link_libraries(asset-nlp PUBLIC asset-pch)
target_precompile_headers(asset-nlp REUSE_FROM asset-pch)
