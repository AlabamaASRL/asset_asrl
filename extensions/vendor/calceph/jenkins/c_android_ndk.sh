#! /bin/bash
#/*-----------------------------------------------------------------*/
#/*! 
#  \file c_android_ndk.sh
#  \brief jenkins tests of the ndk compiler on android. 
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2025, CNRS
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

# caller must define before the variable ANDROID_PLATFORM_VERSION : 
# e.g. export ANDROID_PLATFORM_VERSION=33 # for android Tiramisu

set -e 
set -x

export NDK_VERSION=r28b
export ANDROID_NDK_HOME=$PWD/android-ndk-$NDK_VERSION
export android_toolchain_file=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
export android_abi=x86_64
# export android_abi=arm64-v8a

wget -q https://dl.google.com/android/repository/android-ndk-$NDK_VERSION-linux.zip 
unzip -q android-ndk-$NDK_VERSION-linux.zip 

mkdir build
cd build 
cmake .. \
  -DENABLE_FORTRAN=OFF \
  -DCMAKE_C_FLAGS="-Wall -Wextra" \
  -DCMAKE_TOOLCHAIN_FILE=$android_toolchain_file \
  -DANDROID_ABI=$android_abi \
  -DANDROID_NDK=$ANDROID_NDK_HOME \
  -DANDROID_PLATFORM=android-$ANDROID_PLATFORM_VERSION
make 