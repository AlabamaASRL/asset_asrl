# 1D Interp Table ##############################################################
add_library(bind-itab-1d STATIC BindInterpTable1D.cpp)
target_link_libraries(bind-itab-1d PUBLIC bind-pch)
target_precompile_headers(bind-itab-1d REUSE_FROM bind-pch)

# 2D Interp Table ##############################################################
add_library(bind-itab-2d STATIC BindInterpTable2D.cpp)
target_link_libraries(bind-itab-2d PUBLIC bind-pch)
target_precompile_headers(bind-itab-2d REUSE_FROM bind-pch)

# 3D Interp Table ##############################################################
add_library(bind-itab-3d STATIC BindInterpTable3D.cpp)
target_link_libraries(bind-itab-3d PUBLIC bind-pch)
target_precompile_headers(bind-itab-3d REUSE_FROM bind-pch)

# 4D Interp Table ##############################################################
add_library(bind-itab-4d STATIC BindInterpTable4D.cpp)
target_link_libraries(bind-itab-4d PUBLIC bind-pch)
target_precompile_headers(bind-itab-4d REUSE_FROM bind-pch)
