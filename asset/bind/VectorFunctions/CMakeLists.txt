add_subdirectory(CommonFunctions)

# ArgsSeg ######################################################################

add_library(bind-argseg-1 STATIC BindArgSeg1.cpp)
target_link_libraries(bind-argseg-1 PUBLIC bind-pch)
target_precompile_headers(bind-argseg-1 REUSE_FROM bind-pch)

add_library(bind-argseg-2 STATIC BindArgSeg2.cpp)
target_link_libraries(bind-argseg-2 PUBLIC bind-pch)
target_precompile_headers(bind-argseg-2 REUSE_FROM bind-pch)

add_library(bind-argseg-3 STATIC BindArgSeg3.cpp)
target_link_libraries(bind-argseg-3 PUBLIC bind-pch)
target_precompile_headers(bind-argseg-3 REUSE_FROM bind-pch)

add_library(bind-argseg-4 STATIC BindArgSeg4.cpp)
target_link_libraries(bind-argseg-4 PUBLIC bind-pch)
target_precompile_headers(bind-argseg-4 REUSE_FROM bind-pch)

add_library(bind-argseg-5 STATIC BindArgSeg5.cpp)
target_link_libraries(bind-argseg-5 PUBLIC bind-pch)
target_precompile_headers(bind-argseg-5 REUSE_FROM bind-pch)

# PythonArgParsing #############################################################

add_library(bind-pyargparse STATIC PythonArgParsing.cpp)
target_link_libraries(bind-pyargparse PUBLIC bind-pch)
target_precompile_headers(bind-pyargparse REUSE_FROM bind-pch)

# BulkOperations ###############################################################

add_library(bind-bulkops STATIC BindBulkOperations.cpp)
target_link_libraries(bind-bulkops PUBLIC
  bind-pch
  bind-pyargparse
  asset-dynstack
  asset-dynsum
)
target_precompile_headers(bind-bulkops REUSE_FROM bind-pch)

# FreeFunctions ################################################################

add_library(bind-freefuncs STATIC BindFreeFunctions.cpp)
target_link_libraries(bind-freefuncs PUBLIC bind-pch)
target_precompile_headers(bind-freefuncs REUSE_FROM bind-pch)

# MatrixFunction ###############################################################

add_library(bind-matfunc STATIC BindMatrixFunction.cpp)
target_link_libraries(bind-matfunc PUBLIC
  bind-pch
  asset-dynstack
)
target_precompile_headers(bind-matfunc REUSE_FROM bind-pch)

# VectorFunction ###############################################################

add_library(bind-vf-1 STATIC BindVectorFunction1.cpp)
target_link_libraries(bind-vf-1 PUBLIC bind-pch)
target_precompile_headers(bind-vf-1 REUSE_FROM bind-pch)

add_library(bind-vf-2 STATIC BindVectorFunction2.cpp)
target_link_libraries(bind-vf-2 PUBLIC
  bind-pch
  bind-pyargparse
  asset-dynstack
)
target_precompile_headers(bind-vf-2 REUSE_FROM bind-pch)

add_library(bind-vfs STATIC BindVectorFunctions.cpp)
target_link_libraries(bind-vfs PUBLIC
  bind-pch
  bind-vf-1
  bind-vf-2
  bind-argseg-1
  bind-argseg-2
  bind-argseg-3
  bind-argseg-4
  bind-argseg-5
  bind-freefuncs
  bind-matfunc
  bind-itab-1d
  bind-itab-2d
  bind-itab-3d
  bind-itab-4d
)
