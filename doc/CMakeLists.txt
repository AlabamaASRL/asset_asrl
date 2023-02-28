# Doxygen Stuff
#find_package(Doxygen)


# set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
# set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
# set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
# set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
# set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})
# add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
#                    DEPENDS ${CAT_CUTIFIER_PUBLIC_HEADERS}
#                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
#                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
#                    COMMENT "Generating Doxygen")

# add_custom_target(doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})




################################################################################
# Sphinx Stuff
find_package(Sphinx)





if(NOT DEFINED SPHINX_THEME)
    set(SPHINX_THEME default)
endif()

if(NOT DEFINED SPHINX_THEME_DIR)
    set(SPHINX_THEME_DIR)
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)
#set(SPHINX_PDF_FILE ${SPHINX_BUILD}/index.pdf)

# Configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${BINARY_BUILD_DIR}/conf.py"
    @ONLY)


file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/tutorials/_static/"
DESTINATION  "${SPHINX_BUILD}/tutorials/_static/")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/_static/ASSETLOGO.svg"
DESTINATION  "${SPHINX_BUILD}/_static/")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/examples/_static/"
DESTINATION  "${SPHINX_BUILD}/examples/_static/")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/.nojekyll"
DESTINATION  "${SPHINX_BUILD}/")


message(STATUS ${BINARY_BUILD_DIR})

message(STATUS ${PROJECT_SOURCE_DIR})

message(STATUS ${PROJECT_NAME})

# Run Sphinx
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                     ${SPHINX_EXECUTABLE} -v -v -v -q -b html
                    -c ${BINARY_BUILD_DIR}
                    -d ${SPHINX_CACHE_DIR}
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                   ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                   ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
                   ${CMAKE_CURRENT_SOURCE_DIR}/tutorials
                   ${CMAKE_CURRENT_SOURCE_DIR}/examples
                   ${CMAKE_CURRENT_SOURCE_DIR}/python
                   COMMENT "Generating Sphinx HTML")

add_custom_target(sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})


