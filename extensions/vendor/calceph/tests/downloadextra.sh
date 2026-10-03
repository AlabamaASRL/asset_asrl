#! /bin/sh

#/*-----------------------------------------------------------------*/
#/*! 
#  \file downloadextra.sh
#  \brief Download from internet several kernels to perform extra tests
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2018-2025, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
#  
#*/
#/*-----------------------------------------------------------------*/
#
#/*-----------------------------------------------------------------*/
#/* License  of this file :
#  This file is "triple-licensed", you have to choose one  of the three licenses 
#  below to apply on this file.
#  
#     CeCILL-C
#     	The CeCILL-C license is close to the GNU LGPL.
#     	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#  
#  or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#  
#  or CeCILL v2.1
#       The CeCILL license is compatible with the GNU GPL.
#       ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
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
which wget >/dev/null || exit 1
echo "downloading kernel sat393_daphnis.bsp..."
wget -o /dev/null -O sat393_daphnis.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/sat393_daphnis.bsp 

echo "downloading bc_mcs_mct_50025_20180417_20241218_v01.bsp..."
wget -o /dev/null -O bc_mcs_mct_50025_20180417_20241218_v01.bsp https://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/spk/bc_mcs_mct_50025_20180417_20241218_v01.bsp 

echo "downloading HERA_NomTrajDCP1_v01.bsp..."
wget -o /dev/null -O HERA_NomTrajDCP1_v01.bsp https://spiftp.esac.esa.int/data/SPICE/HERA/kernels/spk/HERA_NomTrajDCP1_v01.bsp

echo "downloading juice_crema_5_0_v01.bsp..."
wget -o /dev/null -O juice_crema_5_0_v01.bsp https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_crema_5_0_v01.bsp

echo "downloading juice_orbc_000082_230414_310721_v01.bsp..."
wget -o /dev/null -O juice_orbc_000082_230414_310721_v01.bsp https://spiftp.esac.esa.int/data/SPICE/JUICE/kernels/spk/juice_orbc_000082_230414_310721_v01.bsp

echo "downloading epm2021.bsp ..."
wget -o /dev/null -O epm2021.bsp https://ftp.iaaras.ru/pub/epm/EPM2021/SPICE/epm2021.bsp

echo "downloading kernel de405unxp2000.405 (sunos)..."
wget -o /dev/null -O unxp2000.405 ftp://ssd.jpl.nasa.gov/pub/eph/planets/SunOS/de405/unxp2000.405
echo "checking Ephemeris file version de405unxp2000.405 ..."
../tools/calceph_inspector unxp2000.405 | grep 'Ephemeris file version' | grep 405 >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel epm2017.tpc..."
wget -o /dev/null -O epm2017.tpc ftp://ftp.iaaras.ru/pub/epm/EPM2017/SPICE/epm2017.tpc
echo "checking Ephemeris file version epm2017.tpc ..."
../tools/calceph_inspector epm2017.tpc | grep 'Ephemeris file version' | grep EPM2017 >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop06c_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop06c_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop06c/inpop06c_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop06c_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop06c_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP06C >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop08a_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop08a_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop08a/inpop08a_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop08a_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop08a_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP08A >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop10a_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop10a_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop10a/inpop10a_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop10a_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop10a_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP10A >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop10b_TDB_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop10b_TDB_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop10b/inpop10b_TDB_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop10b_TDB_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop10b_TDB_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP10B >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop10e_TDB_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop10e_TDB_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop10e/inpop10e_TDB_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop10e_TDB_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop10e_TDB_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP10E >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop13b_TDB_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop13b_TDB_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop13b/inpop13b_TDB_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop13b_TDB_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop13b_TDB_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP13B >/dev/null || { echo 'bad file version' ; exit 1 ; }

echo "downloading kernel inpop13c_TDB_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop13c_TDB_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop13c/inpop13c_TDB_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop13c_TDB_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop13c_TDB_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP13C >/dev/null || { echo 'bad file version' ; exit 1 ; }
 
echo "downloading kernel inpop17a_TDB_m100_p100_bigendian.dat ..."
wget -o /dev/null -O inpop17a_TDB_m100_p100_bigendian.dat https://ftp.imcce.fr/pub/ephem/planets/inpop17a/inpop17a_TDB_m100_p100_bigendian.dat
echo "checking Ephemeris file version inpop17a_TDB_m100_p100_bigendian.dat ..."
../tools/calceph_inspector inpop17a_TDB_m100_p100_bigendian.dat | grep 'Ephemeris file version' | grep INPOP17A >/dev/null || { echo 'bad file version' ; exit 1 ; }
   

echo "downloading kernel inpop10b_TDB_m100_p100_spice.tar.gz ..."
wget -o /dev/null -O inpop10b_TDB_m100_p100_spice.tar.gz https://ftp.imcce.fr/pub/ephem/planets/inpop10b/inpop10b_TDB_m100_p100_spice.tar.gz
tar xzf inpop10b_TDB_m100_p100_spice.tar.gz
echo "checking Ephemeris file version inpop10b_TDB_m100_p100_spice.tpc ..."
../tools/calceph_inspector inpop10b_TDB_m100_p100_spice.tpc | grep 'Ephemeris file version' | grep INPOP10B >/dev/null || { echo 'bad file version' ; exit 1 ; }
rm -f inpop1??_TDB_m100_p100_spice.*

echo "downloading kernel inpop10e_TDB_m100_p100_spice_release2.tar.gz ..."
wget -o /dev/null -O inpop10e_TDB_m100_p100_spice.tar.gz https://ftp.imcce.fr/pub/ephem/planets/inpop10e/inpop10e_TDB_m100_p100_spice_release2.tar.gz
tar xzf inpop10e_TDB_m100_p100_spice.tar.gz
echo "checking Ephemeris file version inpop10e_TDB_m100_p100_spice.tpc ..."
../tools/calceph_inspector inpop10e_TDB_m100_p100_spice.tpc | grep 'Ephemeris file version' | grep INPOP10E >/dev/null || { echo 'bad file version' ; exit 1 ; }
rm -f inpop1??_TDB_m100_p100_spice.*

echo "downloading kernel inpop13b_TDB_m100_p100_spice.tar.gz ..."
wget -o /dev/null -O inpop13b_TDB_m100_p100_spice.tar.gz https://ftp.imcce.fr/pub/ephem/planets/inpop13b/inpop13b_TDB_m100_p100_spice.tar.gz
tar xzf inpop13b_TDB_m100_p100_spice.tar.gz
echo "checking Ephemeris file version inpop13b_TDB_m100_p100_spice.tpc ..."
../tools/calceph_inspector inpop13b_TDB_m100_p100_spice.tpc | grep 'Ephemeris file version' | grep INPOP13B >/dev/null || { echo 'bad file version' ; exit 1 ; }
rm -f inpop1??_TDB_m100_p100_spice.*

echo "downloading kernel inpop13c_TDB_m100_p100_spice.tar.gz ..."
wget -o /dev/null -O inpop13c_TDB_m100_p100_spice.tar.gz https://ftp.imcce.fr/pub/ephem/planets/inpop13c/inpop13c_TDB_m100_p100_spice.tar.gz
tar xzf inpop13c_TDB_m100_p100_spice.tar.gz
echo "checking Ephemeris file version inpop13c_TDB_m100_p100_spice.tpc ..."
../tools/calceph_inspector inpop13c_TDB_m100_p100_spice.tpc | grep 'Ephemeris file version' | grep INPOP13C >/dev/null || { echo 'bad file version' ; exit 1 ; }
rm -f inpop1??_TDB_m100_p100_spice.*

echo "downloading kernel inpop17a_TDB_m100_p100_spice.tar.gz ..."
wget -o /dev/null -O inpop17a_TDB_m100_p100_spice.tar.gz https://ftp.imcce.fr/pub/ephem/planets/inpop17a/inpop17a_TDB_m100_p100_spice.tar.gz
tar xzf inpop17a_TDB_m100_p100_spice.tar.gz
echo "checking Ephemeris file version inpop17a_TDB_m100_p100_spice.tpc ..."
../tools/calceph_inspector inpop17a_TDB_m100_p100_spice.tpc | grep 'Ephemeris file version' | grep INPOP17A >/dev/null || { echo 'bad file version' ; exit 1 ; }
rm -f inpop1??_TDB_m100_p100_spice.*
   
    