#/*-----------------------------------------------------------------*/
#/*! 
#  \file getfov.py
#  \brief Check if calceph_getfov works.
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2025, CNRS
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
import math

from calcephpy import CalcephBin
 
class TestGetFov(unittest.TestCase):
 
    def test_getfov(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_ik.ti"))
        shape, frame, vector, arraybounds = peph.getfov(-42552)
            
        if (shape!=2 or len(arraybounds)!=12 or frame!="EXAMPLE_RECTANGLE") :
            print("shape=", shape)
            print("frame=", frame)
            print("vector=", vector)
            print("arraybounds=", arraybounds)
            raise RuntimeError("invalid shape or len arraybounds")

        expected_rectangle = [0.063657,0.000004,0.997972, 0.063657, -0.000004, 0.997972,  0.063666, -0.000004, 0.997971, 0.063666, 0.000004, 0.997971 ]
        expected_vector = [0.0636614381316129, 0, 0.997971553349531]     
        for j in range(3):
            if (abs(expected_vector[j]-vector[j])>=1E-14):
                print("computed vector=", vector)
                print("expected vector=", expected_vector)
                print("arraybounds=", arraybounds)
                raise RuntimeError("invalid vector")

        for j in range(12):
            if (abs(expected_rectangle[j]-arraybounds[j])>=1E-6):
                print("computed rectangle=", arraybounds)
                print("expected rectangle=", expected_rectangle)
                print("vector=", vector)
                raise RuntimeError("invalid rectangle")

        peph.close()
 
if __name__ == '__main__':
    unittest.main()
