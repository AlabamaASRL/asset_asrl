CALCEPH library
===============           

Description
-----------

This is a release of the  CALCEPH library. This library  is
developed in the "Astronomy and Dynamical System" team
at LTE, Observatoire de Paris, CNRS, PSL Research University, Sorbonne Universite (PARIS).  
The contact email is Mickael.Gastineau@obspm.fr .

This library is available at :  https://calceph.imcce.fr

This library is designed to access and interpolate INPOP and JPL Ephemeris data.
The sources needed to build the CALCEPH library are in the directory src.


The library is "triple-licensed" (CeCILL-C,CeCILL-B or CeCILL),
you have to choose one of the three licenses  below to apply on the library.
  
  CeCILL-C

	The CeCILL-C license is close to the GNU LGPL.
    ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html or COPYING_CECILL_C.LIB)
 
or  
   CeCILL-B
    The CeCILL-B license is close to the BSD.
    (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)

or  
   CeCILL v2.1
    The CeCILL license is compatible with the GNU GPL.
    ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html or COPYING_CECILL_V2.1.LIB)
    
Installation
------------
To compile and use this library, 
please refer to the detailed documentation in the directory "doc" 
or to the section "Installation" of the website https://calceph.imcce.fr

Simplified installation notes for C/Fortran API interface :

 On a Unix-like system (Linux, macOS, BSD, cygwin, MinGW, ...), 
 execute the following statements to install your library in the default directory  /usr/local.
 It creates a temporary folder *build*, which contiains the intermediate files.

  - cmake -S . -B build
  - cmake --build build --target all 
  - cmake --build build --target test 
  - cmake --build build --target install

 On a Windows system with the Microsoft Visual C++ compiler, 
 execute the following statements to install your library in the directory C:\CALCEPH

  - cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX="C:\CALCEPH" -S . -B build
  - cmake --build build --target all 
  - cmake --build build --target test 
  - cmake --build build --target install


Simplified installation notes for python developers with pip :

 It is possible to build and install the latest CALCEPH library with pip

    pip install --user calcephpy
    
 If you have administrative rights on the computer, you can do
    pip install  calcephpy
    
 If you have installed with pip, you can keep your installation up to date by upgrading :
    pip install --user --upgrade calcephpy


Reporting issues
----------------

Report issues using the bugtracker https://mantisbt.imcce.fr/ 