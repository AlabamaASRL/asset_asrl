#/*-----------------------------------------------------------------*/
#/*! 
#  \file c_win64_cl.ps1
#  \brief jenkins tests of the Microsoft Visual C++ compiler (cl.exe) 64 bits. 
#  \author  M. Gastineau 
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
#
#   Copyright, 2023-2024, CNRS
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

Set-Variable -Name "PATHVISUALSTUDIO" -Value "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools"

$env:Path
$env:Path = $env:Path -replace '"', ""
$env:Path

$env:Path += ";$PATHVISUALSTUDIO"; 
& "${env:COMSPEC}" /s /c "`"$PATHVISUALSTUDIO\vsdevcmd.bat`" -no_logo -arch=amd64 -host_arch=amd64  && set" | foreach-object {
    $name, $value = $_ -split '=', 2
    set-content env:\"$name" $value
}

New-Item -path build_static -ItemType "directory" 
Set-Location -Path build_static

$tempDirectory = [System.IO.Path]::GetTempPath();
Set-Variable -Name "CALCEPH_INSTALL" -Value "${tempDirectory}\calceph_install"
Remove-Item ${CALCEPH_INSTALL} -Recurse -ErrorAction Ignore

# static parts
& cmake.exe  -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX="${CALCEPH_INSTALL}" -DCMAKE_C_FLAGS="/wd4996" ..
if ($lastexitcode -ne 0) {exit $lastexitcode}
& cmake.exe --% --build . --target all
if ($lastexitcode -ne 0) {exit $lastexitcode}
$Env:CTEST_OUTPUT_ON_FAILURE=1
& cmake.exe --% --build . --target test
if ($lastexitcode -ne 0) {exit $lastexitcode}
& cmake.exe --% --build . --target install
if ($lastexitcode -ne 0) {exit $lastexitcode}
cd ..


# dynamic parts
New-Item -path build_shared -ItemType "directory" 
Set-Location -Path build_shared
& cmake.exe  -G "NMake Makefiles" -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="${CALCEPH_INSTALL}" -DCMAKE_C_FLAGS="/wd4996" ..
if ($lastexitcode -ne 0) {exit $lastexitcode}
& cmake.exe --% --build . --target all
if ($lastexitcode -ne 0) {exit $lastexitcode}
$Env:CTEST_OUTPUT_ON_FAILURE=1
& cmake.exe --% --build . --target test
if ($lastexitcode -ne 0) {exit $lastexitcode}
& cmake.exe --% --build . --target install
if ($lastexitcode -ne 0) {exit $lastexitcode}
& "${CALCEPH_INSTALL}/bin/calceph_inspector.exe" --% ../examples/example1.bsp
if ($lastexitcode -ne 0) {exit $lastexitcode}

