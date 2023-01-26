
set(CMAKE_VERBOSE_MAKEFILE TRUE)

add_custom_target(pypiwheel  ALL DEPENDS ${CMAKE_SOURCE_DIR}/asset_asrl)
add_dependencies(pypiwheel asset)

set(WHEEL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/asset_asrl)

set(ASSET_BINARY_NAME "asset${PYTHON_MODULE_EXTENSION}")


configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
    @ONLY)


add_custom_command(TARGET pypiwheel 
                    COMMAND ${CMAKE_COMMAND} -E rm -rf ${WHEEL_BUILD_DIR}/
                    COMMENT "Cleaning asset_asrl install directory")


add_custom_command(TARGET pypiwheel 
                    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_SOURCE_DIR}/asset_asrl
                                                ${WHEEL_BUILD_DIR}/asset_asrl
                    DEPENDS ${CMAKE_SOURCE_DIR}/asset_asrl
                    COMMENT "Copied asset_asrl to ${WHEEL_BUILD_DIR}")

add_custom_command(TARGET pypiwheel 
                    COMMAND ${CMAKE_COMMAND} -E copy  ${CMAKE_CURRENT_BINARY_DIR}/setup.py ${WHEEL_BUILD_DIR}
                    
                    COMMENT "Copied setup.py to ${WHEEL_BUILD_DIR}")


add_custom_command(TARGET pypiwheel 
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:asset> ${WHEEL_BUILD_DIR}/
    COMMENT "Copied asset.pyd to ${WHEEL_BUILD_DIR}"
    )




add_custom_command(TARGET pypiwheel 
                    COMMAND
                   ${PYTHON_EXECUTABLE} ${WHEEL_BUILD_DIR}/setup.py bdist_wheel 
                   WORKING_DIRECTORY ${WHEEL_BUILD_DIR}
                   COMMENT "Generating Binary Wheel")


message(STATUS ${ASSET_BINARY_NAME})


