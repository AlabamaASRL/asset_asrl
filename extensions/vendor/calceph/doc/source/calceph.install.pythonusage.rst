Installation
************

The following section describes the installation of the **Python** interface of the library.
If you want to install the interface for another programming language, you have to follow the instructions of the manual of that language.  

Instructions
============

The python interface of the library should be installed using the  package management system **pip** on all operating systems : Windows and Unix-like system (Linux, macOS, BSD, Cygwin, ...).

A python interpreter, compliant at least with Python 3.0 specifications, and the package Cython, setuptools and numpy are required to install the python interface of the library.

Some Linux distributions require the installation development tools of the python software. The name of this package may change on other Linux distributions or operating systems : 

 -  Debian or Ubuntu distributions : package **python3-dev** and **python-dev-is-python3** are required.
 -  Opensuse distribution : package **python3-dev** is required.
 -  Fedora or RedHat distributions : **python3-devel** is required. 


.. highlight::  bash

In addition you need the software *CMake* and a C compiler :

 - On Unix-like operating systems (Linux, MacOS X, ...), you should install gcc or clang. 

 - On windows operating system, you need the Microsoft Visual C++ compiler. 
   If you don't have a C compiler already installed, you can download the
   community edition of visual studio `https://visualstudio.microsoft.com/fr/vs/features/cplusplus/`.
   Before the execution of the next steps, you should execute the following line in the same terminal. You may have to adjust the path according to your version of the Visual studio compiler.


    ::

       "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"




Using pip
~~~~~~~~~


Depending on your python installation, the command **pip** may be replaced by **pip3**. If you use the distribution *anaconda*, you should prefer to use the instruction from the Anaconda section.

The steps are :


- Install the requirements

    ::
    
        pip install Cython setuptools numpy


- Install the library

    ::

        pip install calcephpy
 
 .. highlight::  none





Using Anaconda
~~~~~~~~~~~~~~


Depending on your anaconda installation, the command **pip** may be replaced by **pip3**. 

The steps are :

- Install the gcc compiler from the Anaconda compiler tools :
 
    See the instruction on `https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html`

    For example, on the operating system Linux, it will be :

    ::
    
        conda install gcc_linux-64 cmake make  


- Install the  other requirements

    ::
    
        conda install Cython setuptools numpy


- Install the library

    ::

        pip install calcephpy
 
 .. highlight::  none
