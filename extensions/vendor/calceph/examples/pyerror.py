#/*-----------------------------------------------------------------*/
#/*! 
#  \file pyerror.py 
# \brief Example of usage of the error handler functions with python 
#        This example generates errors.
#        In a first step, on error, it displays a custom message using 
#         its own subroutine.
#        In a second step, on error, the library displays the default error message 
#         but continues the execution.
#        In a last step, on error, the library displays the default error message 
#         but exits.
#        So the message "goodbye" is never displayed.
#
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright,  2016, 2017, CNRS
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
#        The CeCILL-C license is close to the GNU LGPL.
#        ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
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

#/*-----------------------------------------------------------------*/
#/* custom error handler */
#/*-----------------------------------------------------------------*/
def myhandler(msg):
    print("The calceph calls the function myhandler");
    print("The message contains {0} characters\n".format(len(msg)))
    print("The error message is :")
    print("----------------------")
    print(msg)
    print("----------------------")
    print("The error handler returns")

#/*-----------------------------------------------------------------*/
#/* main program */
#/*-----------------------------------------------------------------*/
if __name__ == '__main__':
    jd0=-2442457E16
    dt=0.5E0
    # open the ephemeris file 
    peph = CalcephBin.open("example1.dat")
    print("Step 1 :")
    print("It displays a custom message using its own subroutine\n")
    # set the  error handler to use my own callback 
    seterrorhandler(3, myhandler)
    # call with an invalid time 
    PV = peph.compute(jd0, dt, 10, 3)
    
    print("\n\nStep 2 :")
    print("Calceph library displays the message but it raises an exception\n")
    # set the error handler to display error and continue 
    seterrorhandler(1, 0)
    try:
        # call with an invalid time 
        PV = peph.compute(jd0, dt, 10, 3)
    except:
        print("Python raises an error")
    
    print("\n\nStep 3 :")
    print("calceph library displays the message but the programs exits\n")
    #set the  error handler to stop on error */
    seterrorhandler(2, 0);
    # call with an invalid time
    PV = peph.compute(jd0, dt, 10, 3)
    
    print("Goodbye");
    print("If you see this message, you find a bug !");
    
    # close the ephemeris file 
    peph.close()
