#/*-----------------------------------------------------------------*/
#/*! 
#  \file morientunitfail.py
#  \brief Check the errors of the function calceph_orient_unit.
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2016, 2017, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#*/
#/*-----------------------------------------------------------------*/
# 
#/*-----------------------------------------------------------------*/
#/* License  of this file :
# This file is "triple-licensed", you have to choose one  of the three licenses 
# below to apply on this file.
# 
#    CeCILL-C
#    	The CeCILL-C license is close to the GNU LGPL.
#    	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#   
# or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#  
# or CeCILL v2.1
#      The CeCILL license is compatible with the GNU GPL.
#      ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
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

#/*-----------------------------------------------------------------*/
#/* main program */
#/*-----------------------------------------------------------------*/
import unittest
import openfiles
 
from calcephpy import CalcephBin, Constants
 
class TestOpen(unittest.TestCase):
 
    # set an invalid unit
    def test_morientunitfail(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../examples/example1.dat"))
        jd0=-9.0721875000000000E+03
        dt = jd0-int(jd0)
        jd0 = int(jd0)+2.4515450000000000000E+06
        res = 1
        try:
            PV = peph.orient_unit(jd0, dt, 1, 0)
            print("unit = 0 does not fail")
        except:
            print("normal error")
            res = 0
          
        if res == 1 :
              raise RuntimeError("error not detected");

        try:
            PV = peph.orient_unit(jd0, dt, 301, 0)
            print("unit = 0 does not fail")
        except:
            print("normal error")
            res = 0

        if res == 1 :
              raise RuntimeError("error not detected");
 
        try:
            PV = peph.orient_unit(jd0, dt, 15, Constants.USE_NAIFID, 0)
            print("unit = 0 does not fail")
        except:
            print("normal error")
            res = 0

        if res == 1 :
              raise RuntimeError("error not detected");

if __name__ == '__main__':
    unittest.main()
