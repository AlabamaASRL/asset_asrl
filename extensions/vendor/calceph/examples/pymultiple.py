# /*-----------------------------------------------------------------*/
# /*!
#  \file pymultiple.py
#  \brief Example of usage of the multiple file access functions with the python interface.
#         This example reads the constant EMRAT, AU, GM_Mer and print their values.
#         It computes for a date
#          the geocentric moon coordinates,
#          the value TT-TDB
#          the heliocentric coordinates of Mars.
#
#  \author  M. Gastineau
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris.
#
#   Copyright,  2016-2023, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#
#  History:
# */
# /*-----------------------------------------------------------------*/
#
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

from calcephpy import *

# /*-----------------------------------------------------------------*/
# /* print coordinates */
# /*-----------------------------------------------------------------*/


def printcoord(PV, name):
    print('{0} :\n{1}\n'.format(name, PV))


# /*-----------------------------------------------------------------*/
# /* main program */
# /*-----------------------------------------------------------------*/
if __name__ == '__main__':
    jd0 = 2442457
    dt = 0.5E0
    # open the ephemeris file
    peph = CalcephBin.open("example1.dat")
    print("The ephemeris is already opened\n")

    # print the values of AU, EMRAT and GM_Mer
    AU = peph.getconstant("AU")
    print("AU={0}".format(AU))
    EMRAT = peph.getconstant("EMRAT")
    print("EMRAT={0}".format(EMRAT))
    GM_Mer = peph.getconstant("GM_Mer")
    print("GM_Mer={0}".format(GM_Mer))

    # compute and print the coordinates
    # the geocentric moon coordinates
    moon = peph.getidbyname("Moon", 0)
    earth = peph.getidbyname("Earth", 0)
    PV = peph.compute(jd0, dt, moon, earth)
    printcoord(PV, "geocentric coordinates of the Moon in AU and AU/day")
    PV = peph.compute_unit(jd0, dt, NaifId.MOON, NaifId.EARTH,
                           Constants.UNIT_KM+Constants.UNIT_SEC+Constants.USE_NAIFID)
    printcoord(PV, "geocentric coordinates of the Moon in km and km/s")

    # the value TT-TDB
    PV = peph.compute(jd0, dt, 16, 0)
    print("TT-TDB = {0}\n".format(PV[0]))

    # the heliocentric coordinates of Mars
    PV = peph.compute_unit(jd0, dt, NaifId.MARS_BARYCENTER, NaifId.SUN,
                           Constants.UNIT_AU+Constants.UNIT_DAY+Constants.USE_NAIFID)
    printcoord(PV, "heliocentric coordinates of Mars")

    # print the whole list of the constants
    print("list of constants")
    for j in range(1, peph.getconstantcount()+1):
        nameconstant, valueconstant = peph.getconstantindex(j)
        print("'{0}'\t= {1}".format(nameconstant, valueconstant))

    # close the ephemeris file
    peph.close()
    print("The ephemeris is already closed")
