# /*-----------------------------------------------------------------*/
# /*!
#  \file mcomputeorderarray.py
#  \brief Check the function calceph_compute_order with vectors.
#
#  \author  M. Gastineau
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris.
#
#   Copyright, 2019, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
# */
# /*-----------------------------------------------------------------*/
#
# /*-----------------------------------------------------------------*/
# /* License  of this file :
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

# /*-----------------------------------------------------------------*/
# /* main program */
# /*-----------------------------------------------------------------*/
from calcephpy import *
import openfiles
import unittest
import numpy


class TestOpen(unittest.TestCase):

    # set different size of arrays
    def test_mcompute(self):
        res = 1
        peph = CalcephBin.open(openfiles.prefixsrc(
            "../../examples/example1.dat"))
        jd0 = range(2442457,  2442457+10)
        dt = range(0, 10)
        unit = Constants.UNIT_AU + Constants.UNIT_DAY
        target = 1
        center = 12
        for order in range(4):
            PVlist = peph.compute_order(jd0, dt, target, center, unit, order)
            PVnumpy = peph.compute_order(numpy.asarray(jd0, dtype=numpy.float64), numpy.asarray(
                dt, dtype=numpy.float64), target, center, unit, order)

            if len(PVlist) != 3*(order+1):
                len(PVlist)
                raise RuntimeError("bad size array list")

            if len(PVlist[0]) != 10:
                len(PVlist)
                raise RuntimeError("bad size array list ")

            if len(PVnumpy) != 3*(order+1):
                len(PVnumpy)
                raise RuntimeError("bad size array numpy")

            if len(PVnumpy[0]) != 10:
                len(PVnumpy)
                raise RuntimeError("bad size array numpy")

            for j in range(len(jd0)):
                PV = peph.compute_order(
                    jd0[j], dt[j], target, center, unit, order)
                for k in range(3*(order+1)):
                    if PV[k] != PVlist[k][j]:
                        print("j %d" % j)
                        print("expected :", PV)
                        print("computed :", PVlist[:][j])
                    if PV[k] != PVnumpy[k][j]:
                        print("j %d" % j)
                        print("expected :", PV)
                        print("computed :", PVlist[:][j])


if __name__ == '__main__':
    unittest.main()
