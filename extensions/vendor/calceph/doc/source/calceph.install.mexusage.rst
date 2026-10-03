Installation
************

The following section describes the installation of the **Mex** interface of the library.
If you want to install the interface for another programming language, you have to follow the instructions of the manual of that language.  

Unix-like system (Linux, macOS, BSD, Cygwin, MinGW, ...)
========================================================

Here are the steps needed to install the Mex interface of the library on Unix systems. 
In the following instructions, you must replace */home/mylogin/mydir* by the directory location where you want to install calceph.

*CMake*, available at https://cmake.org, is required.

.. highlight::  bash

    
If you use the Mex interface of the library for Octave (4.0 or later), you have to start Octave and execute the following commands.

    .. parsed-literal::

        pkg install -local  calcephoct-|version|.tar.gz


If you use the Mex interface of the library for Matlab (2017 or later), you have to use a C compiler compliant with your software Matlab, usually gcc.  If you use the gcc compiler, the steps are :

    * Compile of the dynamic library with the following command (replace /home/mylogin/mydir by the correct value) :
    
        .. parsed-literal::

            tar xzf calceph-|version|.tar.gz
            cd calceph-|version|
            mkdir build
            cd build
            cmake -DENABLE_MEX_OCTAVE=ON -DENABLE_FORTRAN=OFF CC=gcc -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir
            make && make test && make install
        
    * Start Matlab and execute (replace /home/mylogin/mydir by the correct value) in order to compile the Mex interface:
        
        .. parsed-literal::

            addpath('/home/mylogin/mydir/libexec/calceph/mex')
            calceph_compilemex()

    * Add the path */home/mylogin/mydir/lib* to the environment variables **LD_LIBRARY_PATH** or **DYLD_LIBRARY_PATH**.
    
    * Add the path */home/mylogin/mydir/libexec/calceph/mex* to the environment variable **MATLABPATH**, in order to have the calceph functions available at the start of Mathlab.

.. warning::
    If you want to use calceph from Matlab on a Mac with M1-chip, you have to change the option comand of configure from **"CC=gcc"** to **CC="gcc -arch x86_64"**.

Windows system
==============

The *cmake* software, available at https://cmake.org, is required.

The instructions and additional requirement, listed in this section, only applies to users of the Microsoft Visual C++ compiler, such as cl.exe.
On other subsystems, such as cygwin or MinGW, no additional requirements are required and the instructions for the Unix operating systems of the previous systems should be followed.


In addition to the Microsoft Visual C++ compiler and *CMake*, you also need the Universal CRT SDK or a Windows SDK. 

The  "Universal CRT (C runtime) SDK" or a "Windows SDK" are now provided with the Microsoft Visual Studio.
You should verify that "Universal CRT (C runtime) SDK" or a "Windows SDK" is selected in the "Visual Studio Installer".      


The steps are :


* Expand the file calceph-|version|.tar.gz

* Execute the command **cmd.exe** from the menu *Start / Execute...*

    This will open a console window

* cd *dir*\\calceph-|version|

    Go to the directory *dir* where |LIBRARYNAME| has been expanded.

* mkdir build 

    This will create the directory build

cd build

    Go to the directory build 


* cmake -DENABLE_MEX_OCTAVE=ON -DCMAKE_INSTALL_PREFIX= *dir* .. 

    The directory *dir* specifies the locaiton where the library will be installed.


* cmake --build . --target all

    This compiles |LIBRARYNAME|.

* cmake --build . --target test

    This will make sure that the |LIBRARYNAME| was built correctly.

    If you get error messages, please report them to |EMAIL| (see :ref:`Reporting bugs`, for information on what to include in useful bug reports).


* cmake --build . --target install

    This will copy the file :file:`calceph.h` to the directory *dir*, the file :file:`libcalceph.lib` to the directory *dir* **\\lib**, the documentation files to the directory *dir* **\\doc**. Note: you need write permissions on these directories.


* If you don't install in a standard path, add  *dir* **\\lib**  to the environment variables **LD_LIBRARY_PATH**.

* Add the path *dir* **\\libexec\\calceph\\mex** to the environment variable **MATLABPATH** 

* Start Matlab or Octave and execute the following command in order to compile the Mex interface:
    
    .. parsed-literal::

       addpath('*dir* **\\libexec\\calceph\\mex**')
       calceph_compilemex()


.. highlight::  none
