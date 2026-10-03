# /*-----------------------------------------------------------------*/
# /*!
#  \file mgetnamebyidss.py
#  \brief Check if calceph_getnamebyidss works.
#
#  \author  M. Gastineau
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris.
#
#   Copyright, 2024, CNRS
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
import unittest
import openfiles

from calcephpy import CalcephBin, Constants


class TestOpen(unittest.TestCase):

    # /*-----------------------------------------------------------------*/
    # /* check the invalid id */
    # /*-----------------------------------------------------------------*/
    def check_failure(self, peph, id,  unit):
        name = peph.getnamebyidss(id, unit)
        if (name is None):
            ret = 0
        else:
            print("failure : find valid name but the id is invalid !!!\n")
            print("name=", name)
            print("unit=",  unit)
            print("id=",  id)
            ret = 1

        return ret


# /*-----------------------------------------------------------------*/
# /* check the valid id */
# /*-----------------------------------------------------------------*/
    def check_valid(self, peph, name_expected, unit, id):
        ret = 0
        try:
            name = peph.getnamebyidss(id, unit)
            if (name_expected != name):
                print("failure : find invalid name but the id is valid !!!\n")
                print("name_expected=", name_expected)
                print("name=", name)
                print("unit=",  unit)
                print("id=",  id)
                ret = 1
        except:
            ret = 1

        return ret


# /*-----------------------------------------------------------------*/
# /* main program */
# /*-----------------------------------------------------------------*/


    def test_getidbyname(self):
        peph = CalcephBin.open(openfiles.prefixsrc(
            "../../examples/example1.tpc"))
        res = 0

        res = res + self.check_failure(peph, -5000, 0)
        res = res + self.check_failure(peph, 998, 0)
        res = res + self.check_failure(peph, 21, 0)
        res = res + \
            self.check_failure(peph, -5000, Constants.USE_NAIFID)
        res = res + self.check_failure(peph, 998, Constants.USE_NAIFID)

        res = res + self.check_valid(peph, "JUPITER BARYCENTER", 0, 5)


        res = res + self.check_valid(peph, "SATURN BARYCENTER",
                                     Constants.USE_NAIFID, 6)

        res = res + self.check_valid(peph, "SATURN",
                                     Constants.USE_NAIFID, 699)

        res = res + self.check_valid(peph, "MERCURY BARYCENTER", 0, 1)
        res = res + self.check_valid(peph, "VENUS BARYCENTER", 0, 2)
        res = res + self.check_valid(peph, "EARTH", 0, 3)
        res = res + self.check_valid(peph, "MARS BARYCENTER", 0, 4)
        res = res + self.check_valid(peph, "SATURN BARYCENTER", 0, 6)
        res = res + self.check_valid(peph, "URANUS BARYCENTER", 0, 7)
        res = res + self.check_valid(peph, "NEPTUNE BARYCENTER", 0, 8)
        res = res + self.check_valid(peph, "PLUTO BARYCENTER", 0, 9)
        res = res + self.check_valid(peph, "MOON", 0, 10)
        res = res + self.check_valid(peph, "SUN", 0, 11)
        res = res + \
            self.check_valid(peph, "SOLAR SYSTEM BARYCENTER", 0, 12)
        res = res + self.check_valid(peph, "EARTH MOON BARYCENTER", 0, 13)

        res = res + self.check_valid(peph, "EARTH",
                                     Constants.USE_NAIFID, 399)
        res = res + self.check_valid(peph, "MOON", Constants.USE_NAIFID, 301)
        res = res + self.check_valid(peph, "SUN", Constants.USE_NAIFID, 10)
        res = res + \
            self.check_valid(peph, "SOLAR SYSTEM BARYCENTER",
                             Constants.USE_NAIFID, 0)
        res = res + \
            self.check_valid(peph, "EARTH MOON BARYCENTER",
                             Constants.USE_NAIFID, 3)

        res = res + self.check_valid(peph, "2 PALLAS",
                                     Constants.USE_NAIFID, 2000002)
        res = res + self.check_valid(peph, "1 CERES",
                                     Constants.USE_NAIFID, 2000001)

        if (res != 0):
            print("res=", res)
            raise RuntimeError("The test fail")

        peph.close()


if __name__ == '__main__':
    unittest.main()
