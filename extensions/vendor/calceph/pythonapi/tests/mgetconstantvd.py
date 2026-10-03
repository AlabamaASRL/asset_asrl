#/*-----------------------------------------------------------------*/
#/*! 
#  \file mgetconstantvd.py
#  \brief Check if calceph_getconstantvd works.
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2016, 2017, 2018, CNRS
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
 
from calcephpy import CalcephBin
 
class TestOpen(unittest.TestCase):
 
    def test_mgetconstant(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/checktpc_11627.tpc"))
        v1 = peph.getconstantvd("BODY000_GMLIST1")
        v1_ref = [699.]
        if (v1!=v1_ref):
             print(cmp(v1,v1_ref))
             print(v1)
             print(v1_ref)
             raise RuntimeError("The test BODY000_GMLIST1 fails")
      
        v2 = peph.getconstantvd("BODY000_GMLIST2")
        v2_ref = [499.,599.]
        if (v2!=v2_ref):
             print(v2)
             raise RuntimeError("The test BODY000_GMLIST2 fails")

        v4 = peph.getconstantvd("BODY000_GMLIST4")
        v4_ref = [199,299,301,399]
        if (v4!=v4_ref):
             print(v4)
             raise RuntimeError("The test BODY000_GMLIST4 fails")

        peph.close()
 
if __name__ == '__main__':
    unittest.main()
