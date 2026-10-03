#/*-----------------------------------------------------------------*/
#/*! 
#  \file pycompute.py 
# \brief simple computation with a file and a date provided on the command line
#  e.g.: python pycompute  myfile.dat 2.451545E+06
#  or .e.g:  python pycompute  myfile.bsp myfile.tpc  2.451545E+06
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2016, 2017, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#
#  History:                                                                
#*/
#/*-----------------------------------------------------------------*/
#
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

from calcephpy import *
import sys

#/*-----------------------------------------------------------------*/
#/* print coordinates */
#/*-----------------------------------------------------------------*/
def printcoord(PV,name):
    print('{0} :\n{1}\n'.format(name,PV))

#/*-----------------------------------------------------------------*/
#/* main program */
#/*-----------------------------------------------------------------*/
if __name__ == '__main__':
    args = sys.argv
    if (len(sys.argv)<3):
        print("Usage : python pycompute.py filename1 filename2 ... filenamen juliandate")
        print("                  filename? : binary or spice ephemeris file names")
        print("                  juliandate : julian date (floating-point number)")
        print("")
        print("e.g.  : python pycompute.py inpop08a_m100_p100_littleendian.dat 2.451545E+06")
        print("e.g.  : python pycompute.py de421.bsp masses.tpc 2.451545E+06");
        sys.exit(1)
        
    jd0 = float(args[len(sys.argv)-1])

    # open the ephemeris file 
    print(args[1:len(sys.argv)-1])
    peph = CalcephBin.open(args[1:len(sys.argv)-1]) 

    #perform the computation
    PV = peph.compute_unit(jd0, 0E0, NaifId.EARTH, NaifId.SUN, 
                           Constants.UNIT_AU+Constants.UNIT_DAY+Constants.USE_NAIFID)
                           
    print("julian date : {0}".format(jd0))
    print("heliocentric coordinates of the Earth in km and km/s")
    print("cartesian heliocentric position of the Earth : {0}".format(PV[0:3]))
    print("cartesian heliocentric velocity of the Earth : {0}".format(PV[3:6]))

    # close the ephemeris file 
    peph.close()
