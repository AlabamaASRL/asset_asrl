include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Solvers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/TypeDefs)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Integrators)

set(ASSET_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/OptimalControl
  ${CMAKE_CURRENT_SOURCE_DIR}/Solvers
  ${CMAKE_CURRENT_SOURCE_DIR}/TypeDefs
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils
  ${CMAKE_CURRENT_SOURCE_DIR}/VectorFunctions
  ${CMAKE_CURRENT_SOURCE_DIR}/Integrators
  PARENT_SCOPE
)

#-------------------------------------------------------------------------------
## PCH
add_library(pch STATIC pch.h pch.cpp  )
target_precompile_headers(pch PRIVATE pch.h 
VectorFunctions/ASSET_VectorFunctions.h)

target_include_directories(pch PRIVATE ${INCLUDE_DIRS})
#target_link_libraries(pch PRIVATE fmt::fmt-header-only)
target_compile_options(pch PUBLIC ${COMPILE_FLAGS})



## Put all ".h"/".cpp" pairs here
# Example: add_library(myODE STATIC odes/myode.h odes/myode.cpp)
if(BUILD_ASSET_ALL )
    add_library(vectorfunctions STATIC VectorFunctions/ASSET_VectorFunctions.h
      VectorFunctions/ASSET_VectorFunctions.cpp
      VectorFunctions/VectorFunctionBuildPart1.cpp
      VectorFunctions/VectorFunctionBuildPart2.cpp
      VectorFunctions/ArgsSegBuildPart1.cpp
      VectorFunctions/ArgsSegBuildPart2.cpp
      VectorFunctions/ArgsSegBuildPart3.cpp
      VectorFunctions/ArgsSegBuildPart4.cpp
      VectorFunctions/ArgsSegBuildPart5.cpp
      VectorFunctions/BulkOperationsBuild.cpp
      VectorFunctions/FreeFunctionsBuild.cpp
      VectorFunctions/MatrixFunctionBuild.cpp
      VectorFunctions/PythonArgParsing.cpp
     )
    target_include_directories(vectorfunctions PRIVATE ${INCLUDE_DIRS})
    target_compile_options(vectorfunctions PUBLIC ${COMPILE_FLAGS})
    target_precompile_headers(vectorfunctions REUSE_FROM pch)
    
endif()



if(BUILD_ASSET_ALL)
  add_library(optimalcontrol STATIC OptimalControl/ASSET_OptimalControl.h
    OptimalControl/ASSET_OptimalControl.cpp
    OptimalControl/ODEPhaseBase.h
    OptimalControl/ODEPhaseBase.cpp
    OptimalControl/GenericODESBuildPart1.cpp
    OptimalControl/GenericODESBuildPart2.cpp
    OptimalControl/GenericODESBuildPart3.cpp
    OptimalControl/GenericODESBuildPart4.cpp
    OptimalControl/GenericODESBuildPart5.cpp
    OptimalControl/GenericODESBuildPart6.cpp
    OptimalControl/PhaseIndexer.h
    OptimalControl/PhaseIndexer.cpp
    OptimalControl/OptimalControlProblem.h
    OptimalControl/OptimalControlProblem.cpp
   )

  target_include_directories(optimalcontrol PRIVATE ${INCLUDE_DIRS})
  target_precompile_headers(optimalcontrol REUSE_FROM pch)

  target_compile_options(optimalcontrol PUBLIC ${COMPILE_FLAGS})
  


endif()



add_library(psiopt STATIC
  Solvers/PSIOPT.h Solvers/PSIOPT.cpp
  Solvers/NonLinearProgram.h Solvers/NonLinearProgram.cpp
  Solvers/Jet.h Solvers/Jet.cpp
  Solvers/OptimizationProblemBase.h Solvers/OptimizationProblemBase.cpp
  Solvers/OptimizationProblem.h Solvers/OptimizationProblem.cpp
 )
target_include_directories(psiopt PRIVATE ${INCLUDE_DIRS})
target_compile_options(psiopt PUBLIC ${COMPILE_FLAGS})



add_library(utils STATIC
  Utils/ColorText.h Utils/ColorText.cpp
  Utils/GetCoreCount.h Utils/GetCoreCount.cpp
  Utils/MemoryManagement.h Utils/MemoryManagement.cpp
 )
target_include_directories(utils PRIVATE ${INCLUDE_DIRS})
target_compile_options(utils PUBLIC ${COMPILE_FLAGS})
target_link_libraries(utils PRIVATE fmt::fmt-header-only)


add_library(astro STATIC
  Astro/ASSET_Astro.h 
  Astro/ASSET_Astro.cpp
  Astro/KeplerUtils.h 
  Astro/KeplerUtils.cpp
  Astro/LambertSolver.h 
  Astro/LambertSolver.cpp
  Astro/LambertSolverMultiRevTest.h 
  Astro/LambertSolverMultiRevTest.cpp
  )


target_include_directories(astro PRIVATE ${INCLUDE_DIRS})
target_compile_options(astro PUBLIC ${COMPILE_FLAGS})



if(BUILD_ASSET_ALL AND NOT LINK_TIME_OPT)
#-------------------------------------------------------------------------------
    ## ASSET Test Executable

    add_executable(asset_tests main.cpp )
    target_compile_options(asset_tests PUBLIC ${COMPILE_FLAGS})

    target_link_libraries(asset_tests PUBLIC pch)
    target_link_libraries(asset_tests PUBLIC psiopt)
    target_link_libraries(asset_tests PUBLIC utils)
    
    target_link_libraries(asset_tests PUBLIC optimalcontrol)
    target_link_libraries(asset_tests PUBLIC vectorfunctions)
    target_link_libraries(asset_tests PUBLIC astro)
    
    target_include_directories(asset_tests PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(asset_tests PRIVATE ${LINK_LIBS})
    if(UNIX)
      set_target_properties(asset_tests PROPERTIES LINK_FLAGS ${LINKER_FLAGS})
    endif()
endif()

#---------------------------------------------------------------------------------




if(BUILD_ASSET_ALL)
    #-------------------------------------------------------------------------------
    ## ASSET Python Module
    pybind11_add_module(asset main.cpp )


    target_compile_options(asset PUBLIC  ${COMPILE_FLAGS})
    target_link_libraries(asset PUBLIC psiopt)
    target_link_libraries(asset PUBLIC utils)

    target_link_libraries(asset PUBLIC optimalcontrol)
    target_link_libraries(asset PUBLIC vectorfunctions)
    target_link_libraries(asset PUBLIC astro)

    target_include_directories(asset PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(asset PRIVATE ${LINK_LIBS})
    #set_target_properties(asset PROPERTIES CXX_VISIBILITY_PRESET hidden)

    if(UNIX)
      set_target_properties(asset PROPERTIES LINK_FLAGS ${LINKER_FLAGS})
    endif()

    #message(STATUS ${LINK_LIBS})
    #message(STATUS ${LINKER_FLAGS})


    add_custom_target(pyassetsrc ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/asset_asrl)

    #add_dependencies(asset pyassetsrc)




    message(STATUS "CMAKE DIR LOC" ${CMAKE_CURRENT_SOURCE_DIR})

    if(PYTHON_LOCAL_INSTALL_DIR)
      add_custom_command(TARGET asset POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:asset> ${PYTHON_LOCAL_INSTALL_DIR}/
        COMMENT "Copied asset.pyd to ${PYTHON_LOCAL_INSTALL_DIR}"
      )
      message(STATUS "Cleaning pyasset install directory")
      #file(REMOVE_RECURSE  ${PYTHON_LOCAL_INSTALL_DIR}/pyasset)

      add_custom_command(TARGET pyassetsrc POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PYTHON_LOCAL_INSTALL_DIR}/asset_asrl
                        COMMENT "Cleaning asset_asrl install directory")

      add_custom_command(TARGET pyassetsrc POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/asset_asrl
                                                    ${PYTHON_LOCAL_INSTALL_DIR}/asset_asrl
                        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/asset_asrl
                        COMMENT "Copied asset_asrl to ${PYTHON_LOCAL_INSTALL_DIR}")

    endif()


    




    #message(STATUS CMAKE_INSTALL_PREFIX)
    
endif()

if(BUILD_ASSET_PROFILE)

    list(APPEND COMPILE_FLAGS "$<$<OR:$<CONFIG:RELEASE>,$<CONFIG:RELWITHDEBINFO>>:/Zi>")
    add_executable(asset_debug littlemain.cpp)
    target_compile_options(asset_debug PUBLIC ${COMPILE_FLAGS})
    target_link_libraries(asset_debug PUBLIC pch)
    target_link_libraries(asset_debug PUBLIC psiopt)
    target_link_libraries(asset_debug PUBLIC utils)

    target_include_directories(asset_debug PRIVATE ${INCLUDE_DIRS})
    target_link_libraries(asset_debug PRIVATE ${LINK_LIBS})
    if(UNIX)
        set_target_properties(asset_debug PROPERTIES LINK_FLAGS ${LINKER_FLAGS})
    endif()
endif()




#-------------------------------------------------------------------------------
## Unit Tests
#add_executable(unit unit.cpp)
#target_compile_options(unit PUBLIC ${COMPILE_FLAGS})
# target_link_libraries(unit PUBLIC pch)
# target_link_libraries(unit PUBLIC vectorfunctions)
#target_include_directories(unit PRIVATE ${INCLUDE_DIRS})
#target_link_libraries(unit PRIVATE ${LINK_LIBS})
#set_target_properties(unit PROPERTIES LINK_FLAGS ${LINKER_FLAGS})
