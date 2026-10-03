#/*-----------------------------------------------------------------*/
#/*! 
#  \file mcomputeunit.py
# \brief Check the output of the function calceph_compute_unit using NaifId.
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
 
from calcephpy import CalcephBin, Constants, NaifId
 
class TestOpen(unittest.TestCase):
 
    def mcheck_main(self, filear, prefetch):
        peph = CalcephBin.open(openfiles.prefixsrc(filear))
        if (prefetch==0):
            peph.prefetch()
        
        AU = peph.getconstant('AU')
        
        fnaifid = open(openfiles.prefixsrc('../../tests/example1_tests_naifid.dat'))
        f = open(openfiles.prefixsrc('../../tests/example1_tests.dat'))
        d = f.readlines()
        datacheck = [x.split() for x in d]
        dnaifid = fnaifid.readlines()
        datachecknaifid = [x.split() for x in dnaifid]
        counter = 0
        for aline in datacheck:
            targetnaifid =  int(datachecknaifid[counter][1])
            centernaifid =  int(datachecknaifid[counter][2])
            counter=counter+1
            jd0 = float(aline[0])
            target =  int(aline[1])
            center = int(aline[2])
            PVcheck = [0,0,0,0,0,0]
            for k in range(0,6):
                PVcheck[k] = float(aline[k+3])
            dt = jd0 - int(jd0)
            jd0 = (int(jd0)) + 2.4515450000000000000E+06
            PV = peph.compute(jd0, dt, target, center)
            PV2 = PV[:]

            seuil = 1E-8
            # /* for libration , return angles between 0 and 2*pi */
            if (target == 15):
                seuil = 1E-7
                j = 2
                pi2 = 6.283185307179586E0
                while (PV[j] >= pi2):
                    PV[j] = PV[j] - pi2
                while (PV[j] <= 0) :
                    PV[j] = PV[j] + pi2

            # /* check with references coordinates */
            n = 6
            for j in range(0,6):
                if (abs(PVcheck[j] - PV[j]) >= seuil):
                    print("failure for ",target, center, " at time ", jd0, dt)
                    print('filear=', filear)
                    print('aline=', aline)
                    for k in range(0,n):
                        print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                    print("prefetch=", prefetch)
                    raise RuntimeError("Internal failure")
                    
            # change target and center using naifid
            if (target==1) and (targetnaifid!=NaifId.MERCURY_BARYCENTER):
                    raise RuntimeError("Invalid NAIFID")
            if (target==2) and (targetnaifid!=NaifId.VENUS_BARYCENTER):
                    raise RuntimeError("Invalid NAIFID")
            if (target==4) and (targetnaifid!=NaifId.MARS_BARYCENTER):
                    raise RuntimeError("Invalid NAIFID")
            if (target==13) and (targetnaifid!=NaifId.EARTH_MOON_BARYCENTER):
                    raise RuntimeError("Invalid NAIFID : EARTH_MOON_BARYCENTER")
            if (target==3) and (targetnaifid!=NaifId.EARTH):
                    raise RuntimeError("Invalid NAIFID : EARTH")

            seuil = 3E-15
            if (target != 15 and target != 16):
                #/* check UA/DAY */
                unit = Constants.UNIT_AU + Constants.UNIT_DAY + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.compute_unit(jd0, dt, targetnaifid, centernaifid, unit)
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV[j]) >= seuil):
                        print("failure AU/DAY for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")

                #/* check UA/SEC */
                unit = Constants.UNIT_AU + Constants.UNIT_SEC + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.compute_unit(jd0, dt, targetnaifid, centernaifid, unit)
                for j in range(3,6):
                    PVcheck[j] = PVcheck[j]*86400E0
                
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV[j]) >= seuil):
                        print("failure AU/SEC for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")

                #/* check KM/DAY */
                seuil = 3E-14
                unit = Constants.UNIT_KM + Constants.UNIT_DAY + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.compute_unit(jd0, dt, targetnaifid, centernaifid, unit)
                for j in range(0,6):
                    PVcheck[j] = PVcheck[j]/AU
                
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV[j]) >= seuil):
                        print("failure KM/DAY for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")
           
                #/* check KM/SEC */
                seuil = 3E-14
                unit = Constants.UNIT_KM + Constants.UNIT_SEC + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.compute_unit(jd0, dt, targetnaifid, centernaifid, unit)
                for j in range(0,6):
                    PVcheck[j] = PVcheck[j]/AU
                for j in range(3,6):
                    PVcheck[j] = PVcheck[j]*86400E0
               
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV[j]) >= seuil):
                        print("failure KM/SEC for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV[k], PVcheck[k], PV[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")

            if (target == 15):
                #/* check RAD/DAY */
                unit = Constants.UNIT_RAD + Constants.UNIT_DAY + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.orient_unit(jd0, dt, NaifId.MOON, unit)
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV2[j]) >= seuil):
                        print("failure RAD/DAY for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV2[k], PVcheck[k], PV2[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")

                #/* check RAD/SEC */
                unit = Constants.UNIT_RAD + Constants.UNIT_SEC + Constants.USE_NAIFID
                n = 6
                PVcheck = peph.orient_unit(jd0, dt, NaifId.MOON, unit)
                for j in range(3,6):
                    PVcheck[j] = PVcheck[j]*86400E0
                
                for j in range(0,6):
                    if (abs(PVcheck[j] - PV2[j]) >= seuil):
                        print("failure RAD/SEC for ",targetnaifid, centernaifid, " at time ", jd0, dt)
                        print('filear=', filear)
                        print('aline=', aline)
                        for k in range(0,n):
                            print(k, "diff= ",target, center, " at time jd0, dt", PVcheck[k] - PV2[k], PVcheck[k], PV2[k])
                        print("prefetch=", prefetch)
                        raise RuntimeError("Internal failure")

        f.close()
        peph.close()
   
    def test_mcheck(self):
        self.mcheck_main("../../examples/example1.dat",0)
        self.mcheck_main("../../examples/example1.dat",1)
        listspk = ["../../examples/example1.bsp", "../../examples/example1.tpc", "../../examples/example1.tf", "../../examples/example1.bpc", "../../examples/example1spk_time.bsp"]
        self.mcheck_main(listspk, 0)
        self.mcheck_main(listspk, 1)
 
if __name__ == '__main__':
    unittest.main()
