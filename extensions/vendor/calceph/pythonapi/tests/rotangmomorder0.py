#/*-----------------------------------------------------------------*/
#/*! 
#  \file rotangmomorder0.py
# \brief Check the output of the function calceph_rotangmom_order for the order 0.
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2017, CNRS
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
 
    def mcheck_main(self, filear, prefetch):
        peph = CalcephBin.open(openfiles.prefixsrc(filear))
        if (prefetch==0):
            peph.prefetch()
        
        f = open(openfiles.prefixsrc('../../tests/example2_rotangmom_tests.dat'))
        d = f.readlines()
        datacheck = [x.split() for x in d]
        for aline in datacheck:
            jd0 = float(aline[0])
            target =  int(aline[1])
            center = int(aline[2])
            PVcheck = [0,0,0]
            for k in range(0,3):
                PVcheck[k] = float(aline[k+3])
            dt = jd0 - int(jd0)
            jd0 = (int(jd0)) + 2.4515450000000000000E+06

            seuil = 3E-10
            #/* check DAY */
            unit = Constants.UNIT_DAY
            if (target>=100):
                unit = unit + Constants.USE_NAIFID
            
            n = 3
            PV = peph.rotangmom_order(jd0, dt, target, unit, 0)
            for j in range(0,n):
                if (abs(PVcheck[j] - PV[j]) >= seuil):
                    print("failure DAY for ",target, center, " at time ", jd0, dt)
                    print('filear=', filear)
                    print('aline=', aline)
                    for k in range(0,n):
                        print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                    print("prefetch=", prefetch)
                    raise RuntimeError("Internal failure")

            #/* check SEC */
            unit = Constants.UNIT_SEC
            if (target>=100):
                unit = unit + Constants.USE_NAIFID
            
            PV = peph.rotangmom_order(jd0, dt, target, unit, 0)
            for j in range(0,n):
                PVcheck[j] = PVcheck[j]/86400E0
            
            for j in range(0,n):
                if (abs(PVcheck[j] - PV[j]) >= seuil):
                    print("failure SEC for ",target, center, " at time ", jd0, dt)
                    print('filear=', filear)
                    print('aline=', aline)
                    for k in range(0,n):
                        print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                    print("prefetch=", prefetch)
                    raise RuntimeError("Internal failure")


        f.close()
        peph.close()
   
    def test_mcheck(self):
        self.mcheck_main("../../examples/example2_rotangmom.dat",0)
        self.mcheck_main("../../examples/example2_rotangmom.dat",1)
 
if __name__ == '__main__':
    unittest.main()
