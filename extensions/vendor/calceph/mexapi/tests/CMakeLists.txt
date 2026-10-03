#/*-----------------------------------------------------------------*/
#/*! 
#  \file CMakeLists.txt
#  \brief tests of the mex api for cmake. 
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2023-2026, CNRS
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


set(CALCEPHMEXSHAREDPATH "${PROJECT_BINARY_DIR}/src")

configure_file(openfilesbinary.m.in openfilesbinary.m)

#####################################################################
# single mex test file
function(buildtestsoctave  sourcename)
  set(TEST_FILE_OCTAVE "${sourcename}")
  configure_file(testoctave.m.in ${sourcename}.m)
  add_test(NAME ${sourcename}_m COMMAND "${OCTAVE_EXECUTABLE}" "-W" "${CMAKE_CURRENT_BINARY_DIR}/${sourcename}.m" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(${sourcename}_m PROPERTIES RUN_SERIAL True) 
endfunction(buildtestsoctave)

set (LIST_TEST_OCTAVE_SIMPLE  mgetconstant           mgetconstantindex  
  getorientrecordindex          morientunitfail      tgetversionstr 
  getpositionrecordindex        topen1               mcheck                                    
  mcomputeorder01              topen2                topen3 
  mcomputeorder23                                    topenfail 
  mcomputeunit                  rotangmomorder0               ttimescale 
  mcomputeunitfail              rotangmomunit                 ttimespan 
  mcomputeunitnaifid            terror                        mcomputearray 
  mcomputeunitarray             mcomputeorderarray            morientunitarray 
  morientorderarray             mrotangmomunitarray           mrotangmomorder0array 
  mgetconstantvd               mgetconstantvs                 tfileversion 
  tisthreadsafe                maxsupportedderivativeorder    mgetidbyname
  mgetnamebyidss               ttime_jd_to_cal      ttime_cal_to_jd
  ttime_jd_tdb_to_cal_utc      ttime_cal_utc_to_jd_tdb
  ttime_jd_tt_to_jd_tdb        ttime_str_any_to_jd_tdb ttime_str_to_jd
  ttime_str_utc_to_jd_tdb      tgetfov
  ttime_jd_tcb_to_jd_tdb       ttime_str_spacecraft_clock_to_jd_tdb ttime_jd_tdb_to_str_spacecraft_clock
)
 
if (Octave_FOUND)
  foreach (OCTAVE_SIMPLE IN LISTS LIST_TEST_OCTAVE_SIMPLE)
      buildtestsoctave(${OCTAVE_SIMPLE})
  endforeach()
  set_tests_properties(mcomputeorder23_m PROPERTIES DEPENDS cmcomputeorder23) 
endif()
