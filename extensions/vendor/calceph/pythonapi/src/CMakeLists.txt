#/*-----------------------------------------------------------------*/
#/*! 
#  \file CMakeLists.txt
#  \brief build chain of the python api for cmake. 
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2021-2023, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#  
#*/
#/*-----------------------------------------------------------------*/
#
#/*-----------------------------------------------------------------*/
#/* License  of this file :
#  This file is "triple-licensed", you have to choose one  of the three licenses 
#  below to apply on this file.
#  
#     CeCILL-C
#     	The CeCILL-C license is close to the GNU LGPL.
#     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#   
#  or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#  
#  or CeCILL v2.1
#       The CeCILL license is compatible with the GNU GPL.
#       ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
#  
# 
# This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under 
# French law and abiding by the rules of distribution of free software.  
# You can  use, modify and/ or redistribute the software under the terms 
# of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA  
# at the following URL "http://www.cecill.info". 
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
# */
# /*-----------------------------------------------------------------*/

# detect if cython supports noexcept
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import Cython; import sys; sys.exit(1 if (Cython.__version__ >= \"3.0.0\") else 0)"
    RESULT_VARIABLE  HAVE_CYTHON_NOEXCEPT
    OUTPUT_QUIET
)

SET(CYTHON_NOEXCEPT noexcept) 
CONFIGURE_FILE(calcephpy.pyx.in calcephpy_after_3_0.pyx @ONLY)
SET(CYTHON_NOEXCEPT) 
CONFIGURE_FILE(calcephpy.pyx.in calcephpy_prev_3_0.pyx @ONLY)


# for the python tests only
if (ENABLE_PYTHON_TESTS)

  FILE(COPY calcephpy.pxd DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

  if(HAVE_CYTHON_NOEXCEPT)
    message(STATUS "Cython supports the keyword noexcept")
    SET(CYTHON_NOEXCEPT noexcept) 
  else() 
    message(STATUS "Cython doesn't support the keyword noexcept")
    SET(CYTHON_NOEXCEPT)
  endif()
  

  CONFIGURE_FILE(calcephpy.pyx.in calcephpy.pyx @ONLY)

  CONFIGURE_FILE(setup.py.in setup.py @ONLY)

  add_custom_target( calcephpy_c  ALL 
    COMMAND "${Python_EXECUTABLE}" setup.py build_ext -i
    DEPENDS setup.py calcephpy.pyx calcephpy.pxd calceph
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM)

endif()

