Installation
************

The following section describes the installation of the **C** and **Fortran** interface of the library.
If you want to install the interface for another programming language, you have to follow the instructions of the manual of that language.  

Quick instructions
==================

Here are the quick steps needed to install the library on Unix systems. 
In the following instructions, you must replace */home/mylogin/mydir* by the directory location where you want to install calceph.


.. highlight::  bash        


If you use the gnu gcc and gfortran compilers, the steps are :

    .. parsed-literal::

        tar xzf calceph-|version|.tar.gz
        cd calceph-|version|
        mkdir build
        cd build
        CC=gcc FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir ..
        cmake --build . --target all 
        cmake --build . --target test 
        cmake --build . --target install 


If you use the Intel c++ and fortran compilers, the steps are :

    .. parsed-literal::

        tar xzf calceph-|version|.tar.gz
        cd calceph-|version|
        mkdir build
        cd build
        CC=icc FC=ifort cmake -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir ..
        cmake --build . --target all 
        cmake --build . --target test 
        cmake --build . --target install 

If you use the llvm clang and flang compilers, the steps are :

    .. parsed-literal::

        tar xzf calceph-|version|.tar.gz
        cd calceph-|version|
        mkdir build
        cd build
        CC=clang FC=flang cmake -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir ..
        cmake --build . --target all 
        cmake --build . --target test 
        cmake --build . --target install 

If you use the Microsoft Visual C++ compilers using NMake generator (recommended method), the steps are :

    .. parsed-literal::

        tar xzf calceph-|version|.tar.gz
        cd calceph-|version|
        mkdir build
        cd build
        cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir ..
        cmake --build . --target all 
        cmake --build . --target test 
        cmake --build . --target install 

If you use the Microsoft Visual C++ compilers using Visual Studio generator, the steps are :

    .. parsed-literal::

        tar xzf calceph-|version|.tar.gz
        cd calceph-|version|
        mkdir build
        cd build
        cmake  -DCMAKE_INSTALL_PREFIX=/home/mylogin/mydir ..
        cmake --build . --target BUILD_ALL
        cmake --build . --target INSTALL
        
        
Requirements
============

You need the following software to build the library 

  - *CMake*, available at https://cmake.org
  - a C compiler, such as *gcc* or *clang* or the Microsoft Visual C++ compiler. 
  - a Fortran compiler,such as *gfortran* or *ifort*, compliant with the Fortran 77, respectively 2003, specifications is required to compile the fortran-77/90/95, respectively fortran-2003, interface of the library.

The additional tools are required to build the documentation

  - the python packages : sphinx, six, sphinx-fortran, sphinx-rtd-theme, sphinxcontrib-matlabdomain. They can be installed using the command pip3 
    
    .. parsed-literal::
      
        pip3 install Sphinx six sphinx-fortran sphinx-rtd-theme sphinxcontrib-matlabdomain

  - pdflatex, available with the software LaTeX, if PDF documentation is requested (-DENABLE_PDF=ON).


**Additional requirements for the Microsoft Visual C++ compiler**

In addition to the Microsoft Visual C++ compiler and *CMake*, you also need the Universal CRT SDK or a Windows SDK. 

The  "Universal CRT (C runtime) SDK" or a "Windows SDK" are now provided with the Microsoft Visual Studio.
You should verify that "Universal CRT (C runtime) SDK" or a "Windows SDK" is selected in the "Visual Studio Installer". 
 

Detailed instructions
=====================


Here are the detailed steps needed to install the library :

* tar xzf calceph-|version|.tar.gz
* cd calceph-|version|
* mkdir build 
* cd build

If you are on a Unix-like system :

* cmake ..

If you are using the Microsoft Visual C++ compiler using the NMake generator (recommended method):

* cmake -G "NMake Makefiles" ..


    Running *cmake* might take a while.  While running, it prints some
    messages telling which features it is checking for.

    *cmake* recognizes the following options to control how it
    operates. The options must specified just after the name of software cmake  : *cmake <options> ..*  .

     * | -DENABLE_FORTRAN={ON|OFF}
       | Enable or disable the fortran-77 and fortran-2003 interface. The default is *ON*. 
     * | -DENABLE_THREAD={ON|OFF}
       | Enable or disable the thread-safe version of the functions |calceph_sopen| and |calceph_scompute|, ... and concurrent access to the function |calceph_compute|, ....  The default is *ON*.
     * | -DBUILD_SHARED_LIBS={ON|OFF}
       | Enable or disable the generation of the shared library, otherwise static library is build. The default is *OFF*, static library is built by default.
     * | -DENABLE_DOC={ON|OFF}
       | Enable or disable the generation of the html documentation and its installation. The default is *OFF*.
     * | -DENABLE_PDF={ON|OFF}
       | Enable or disable the generation of the PDF documentation and its installation. The default is *OFF*.
     * | -DCMAKE_INSTALL_PREFIX= *dir*
       | Use *dir* as the installation prefix.  See the command *make install* for the installation names.

    The default compilers could be changed using the variable CC for C compiler and FC for the Fortran compiler. The default compiler flags could be changed using the variable CFLAGS for C compiler and FCFLAGS for the Fortran compiler.

    .. note::  The option *-DENABLE_PYTHON...* or *-DENABLE_MEX_OCTAVE...* should not be used. They are reserved for the installation of the python  or mex interface.

 * cmake --build . --target all
 
    This compiles the |LIBRARYNAME| in the working directory.

 * cmake --build . --target test

    This will make sure that the |LIBRARYNAME| was built correctly.

    If you get error messages, please report them to |EMAIL| (see :ref:`Reporting bugs`, for information on what to include in useful bug reports).

 * cmake --build . --target doc
 
    This builds the documentation of the |LIBRARYNAME| if ENABLE_DOC was set to ON.


 * cmake --build . --target install

    This will copy the files :file:`calceph.h`, :file:`calceph.mod` and  :file:`f90calceph.h` to the directory **/usr/local/include**, the file  :file:`libcalceph.a`,  :file:`libcalceph.so` to the directory **/usr/local/lib**, and the documentations files to the directory **/usr/local/doc/calceph/** (or if you passed the *-DCMAKE_INSTALL_PREFIX=* option to *cmake*, using the prefix directory given as argument to *-DCMAKE_INSTALL_PREFIX=* instead of  **/usr/local**). Note: you need write permissions on these directories.


Other *make* Targets

    There are some other useful make targets:


    * *clean*

        Delete all object files and archive files, but not the configuration files.


.. hint::

    iI the generation of the shared library is enabled (BUILD_SHARED_LIBS=ON) and you are using the default *Visual Studio* generator with cmake, then the target RUN\_TESTS cannot be executed. 


.. highlight::  none
