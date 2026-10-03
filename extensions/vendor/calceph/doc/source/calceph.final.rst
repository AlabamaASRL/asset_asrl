Release notes
=============

 * Version 5.0.1
    | Support the computation of the acceleration and jerk for the segment 1 and 21 in the SPICE kernel file.
    | Check the Fortran/C Compiler compatibility and emit a cmake warning during the compilation.
    | Fix the compilation if the Fortran compiler does not add a trailing underscore.
    | Fix the compilation if the C compiler supports the ISO C11 multi-threading standard on windows.
    | Fix the compilation with cmake and Visual Studio generators.
    | Fix the compilation if ENABLE_INSTALL=OFF or ENABLE_THREAD=OFF.
    | Remove compilation warnings about missing prototypes, duplicated declarations and implicit conversions.
    | Fix the reduction of the stack size of the arrays for the evaluation with C99 compilers.

 * Version 5.0.0
    | Support the Leapseconds Clock Kernel files (spice files .tls).
    | Support the Scpacecraft Clock Kernel files (spice files .tsc).
    | Support the Instrument Kernels files (spice files .ik).
    | Add the set of functions to convert dates (julian days or calendar dates) between different timescales  (TCB, TDB, TT, TAI, UTC) : calceph_time... .
    | Add the set of functions to write binary SPK and PCK SPICE kernel ephemeris files in TDB or TCB timescales.
    | Add the calceph_getfov to retrieve data about the field of view of the instruments.
    | Add the documentation of the deprecated functions : Single file access functions.
    | Add a new define for asteroids on 8-digit NAIF ID : CALCEPH_ASTEROID_8.
    | Fix the compilation with android-ndk if API level >=23.
    | Fix the returned value of calceph_gettimescale with SPICE kernel files containing segments of type 102.
    | Fix a buffer overflow in the function calceph_getfileversion in the Fortran 2003 interface. Change the array size of the function calceph_getfileversion from the value CALCEPH_MAX_CONSTANTNAME to CALCEPH_MAX_CONSTANTVALUE in the header file and documentation, to conform to the examples and to other interfaces. Only Fortran 2003 are affected by this change : previous header file and documentation contains an invalid declaration of the function.
    | Fix a buffer overflow in the function calceph_getconstantss in the Fortran 2003 interface. Change the array size of the last argument of the function calceph_getconstantss from the value CALCEPH_MAX_CONSTANTNAME to CALCEPH_MAX_CONSTANTVALUE in the header file and documentation, to conform to the examples and to other interfaces. Only Fortran 2003 are affected by this change : previous header file and documentation contains an invalid declaration of the function.
    | Fix the documentation of the functions f90calceph_getconstantss and f90calceph_getconstantvs in the Fortran 77/90/95 interface. Change the array size of the last argument of the function f90calceph_getconstantss from the value CALCEPH_MAX_CONSTANTNAME to CALCEPH_MAX_CONSTANTVALUE, to conform to the examples and to other interfaces. Only Fortran 77/90/95 are affected by this change : possible buffer overflow occurs with previous documentation.
    | Fix the documentation and the behavior with numbers of the function calceph_getconstantvs and calceph_getconstantss.
    | Fix the compilation with vcpkg on the operating system arm64/linux.
    | Remove the undocumented function calceph\_ and calcephinit\_.

 * Version 4.0.5
    | Support the segment 19 in the SPICE kernel file.
    | Fix the installation with Cython>=3.1.0.
    | Fix the compilation of the documentation with Cmake 4.0 or later.

 * Version 4.0.4
    | Fix the calceph_gettimespan function result with non-integer time bounds in SPICE kernel files.
    | Support for parallel execution of the python and octave tests.

 * Version 4.0.3
    | Fix a crash of the function calceph_open and calceph_open_array with some invalid SPICE kernel files (files .tpc and .tf) on arm32v6 processors.
    | Add the man pages for the tools.
    | Replace the calls to sprintf by the calls to the fortified function snprintf.
    | Version 4.0.2 skipped due to an archive problem.

 * Version 4.0.1
    | Fix the function calceph_open and calceph_open_array with some SPICE kernel files (files .bsp and .spk) if the segment descriptor does not contain any segments.

 * Version 4.0.0
    | Use cmake to compile the library. Binary ABI may be broken by this major change in the compilation process. The library remains compatible at the source level: the existing application therefore only needs a recompilation.
    | Add the function calceph_getidbyname and calceph_getnamebyidss.
    | Fix potential crash of fortran-90 interface only, on 32-bit OS, if ephemeris file does not exist.
    | Support frame name with any character except tab, space, comma, parenthese, equal, in the frame SPICE kernel file (files .tf).
    | Fix system error messages to be thread-safe (remove strerror).
    | Fix memory buffer overflow if the number of constants is greater than 400 in the original JPL files.

 * Version 3.5.5
    | Fix memory leaks when files are closed.
    | Fix heap buffer overflow (read only operation) with text SPICE kernel files.
    | Fix memory crash if the SPICE kernel files contain segments with a size of the directory greater than 100.

 * Version 3.5.4
    | Fix the documentation about the unsupported segment 19.
    | Fix the window size if requested time is near a boundary for SPICE kernel files containing segments of type 18, compliant with the SPICE specifications.
    | Fix incorrect results for SPICE kernel files containing segments of type 20.
    | Support the segment 14 in the SPICE kernel file.

 * Version 3.5.3
    | Fix compilation error with Cython 3.0 or later.
    | Fix compilation error with Python 3.10 or later on some linux distributions.
    | Fix the tests suite with octave, due to a change of the syntax parsing by octave.
 
 * Version 3.5.2
    | Fix a possible buffer overflow in the python interface of getfileversion.
    | Fix the undefined external symbol rint with visual C compiler 10.0.
    | Fix the redefinition of vasprintf with the environnment MinGW.

 * Version 3.5.1
    | Support the incremental assignment in the text SPICE kernel file (files .tpc).
    | Support the relative frames in the frame SPICE kernel file (files .tf).
    | Fix the returned values of calceph_getconstantvs with large SPICE kernels.
    | Fix random crash or double-free error when meta-kernel SPICE kernel files are closed (files .mk).
    | Raise an error inside the function calceph_compute if center != 0 for the target (14,15,16,17), to conform to the documentation.
    | Fix the object name Megaclite (519) from NAIFID_MAGACLITE to NAIFID_MEGACLITE.
    
 * Version 3.5.0
    | Add the function calceph_getmaxsupportedorder, calceph_getpositionrecordindex2 and calceph_getorientrecordindex2.
    | Update config.sub and config.guess to support new processors.
    | Fix an incorrect result of calceph_(s)getconstant and calceph_getconstantvd with SPICE text kernel 
    | if the current locale has a decimal separator different than the decimal point (e.g. comma).
    | Fix compilation warnings with gcc 11 about pointers and arrays.
    | Fix the installation with python 3.10 or later (PEP 632).
    | Fix compilation errors with the android SDK API level < 24.
    | Reduce the stack size of the arrays for the evaluation with C99 compilers.
    | Change the array size of the function calceph_getfileversion from the value CALCEPH_MAX_CONSTANTNAME
    | to CALCEPH_MAX_CONSTANTVALUE, to conform to the documentation and examples.
    | Only C and Fortran 2003 are affected by this change : previous header file contains an invalid declaration of the function.
    | Few programs may be affected by this change, as the required array has a smaller size : No buffer overflow may appear. 

 * Version 3.4.7
    | Fix a decode error of the little-endian SPICE kernel files on the big-endian architectures (e.g. processor s390x).
    | Fix the transmission of the flags FCFLAGS to the fortran compilers.

 * Version 3.4.6
    | Fix a wrong error message about unsupported order for the segment 21.
    | Fix incorrect results for SPICE kernel files containing segments of type 21 with many records (>=100) and improved the accuracy if segments of type 21 contain few records (<100).

 * Version 3.4.5
    | Fix a random crash of calceph_open_array if one of the file is invalid.
    | f90calceph_seterrorhandler now ignores the parameter userfunc, instead of the requirement to set to 0, if the parameter type is 1 or 2. userfunc can be an empty function. It fixes compilation errors with gcc 10.1.

 * Version 3.4.4
    | Fix a regression introduced in 3.4.3 (remove a recursion with SPICE kernel files).

 * Version 3.4.3
    | Remove a recursion to read the segments of the SPICE kernel files. It reduces the usage of the stack. 
    | Fix the installation of python package under Anaconda.

 * Version 3.4.2
    | Add a missing makefile for windows system using the Visual C++ compiler. 
    | Support SPICE kernels larger than 4GBytes. 

 * Version 3.4.1
    | Improve the execution time of calceph_open and calceph_open_array if the spice kernels contains a large number of bodies.
    | Update config.sub and config.guess to support arm processors.

 * Version 3.4.0
    | Add the function calceph_isthreadsafe.
    | Multiple threads can now access the same ephemeris descriptor if the function calceph_isthreadsafe returns 1. 
    | Fortran and C examples (f2003parallel.f, cparallel.c), written using OpenMP, are available in the folder examples.
    | Fix an error if multiple SPICE kernels are loaded for the same objects over different time-span.
    | Fix the MinGW Makefiles if the variable MAKE contains spaces.
    | Support the segment 5 and 18 in the SPICE kernel file.
    | Support the euler angles for the orientation stored in a text PCK files (BODY..._POLE_RA, BODY..._POLE_DE, BODY..._POLE_PM, BODY..._NUT_PREC_...).
    | Support the frame 17 (ECLIPJ2000) in the SPICE kernel file.
    | Add the utilities calceph_queryposition and calceph_queryorientation.

 * Version 3.3.1
    | Fix the installation with python 3.7.0 or later.
    | Fix the installation with python and pip on Windows operating system.
    | Add the missing file pythonapi/src/Makefile.mingw for the environnment MinGW.

 * Version 3.3.0 
    | Add the functions calceph_getfileversion.
    | Fix a regression to open some old JPL DE format files.
    | Fix a compiler warning in the file util.c.
    | Support the segments 8, 9, 17 and 21 in the SPICE kernel file.
    | Check the validity of the number of constants in the original INPOP/DE files.
    | For the Python interface, the functions compute??? and orient??? supports now a list or numpy's array for the time parameters.

 * Version 3.2.0 
    | Fix the creation of the dynamic library with msys/mingw on Windows.
    | Fix the returned value of the functions f90calceph_getconstantvd and f90calceph_getconstantvs.
    | Fix a compilation warning with the GNU C compilers 8.0 or later.
    | Support the original JPL files with TT-TDB or with a large number of constants.
    | Support the IAU 1980 Nutation Angles of the JPL files.
    | Add the NAIF identification numbers for DIA, KERBEROS, STYX and SIDING SPRING. 
    | Add the option installnodoc to the make command.

 * Version 3.1.0 
    | Add the Mex interface compliant with Octave 4.0+ and Matlab 2017+.
    | Add the functions calceph_getconstantsd, calceph_getconstantvd and calceph_getconstantss and calceph_getconstantvs.
    | Fix a compilation problem with MinGW if the terminal cmd.exe is used.
    | Fix a wrong function name open_array instead of open in the documentation of the Python interface.
    | Fix the return value of the functions calceph_orient_xxx when the unit CALCEPH_UNIT_RAD is not provided.
    | The return value of the function calceph_(s)getconstant(index) is the number of values associated to the constant.
    | Display a better message for the unsupported old spice kernel (NAIF/DAF)

 * Version 3.0.0 
    | Update the license CeCILL v2.0 to CeCILL v2.1.
    | Fix a decode error for SPICE kernels with a big-endian format.
    | Add the function calceph_gettimescale and calceph_gettimespan.
    | Add the function calceph_getpositionrecordcount and calceph_getpositionrecordindex.
    | Add the function calceph_getorientrecordcount and calceph_getorientrecordindex.
    | Add the function calceph_sgettimescale and calceph_sgettimespan.
    | Support INPOP file format 3.0 (add angular momentum due to the rotation in the binary file).
    | Use sphinx-doc to produce the documentation.

 * Version 2.3.2
    | Fix the return value of the function calceph_getconstant if the constant name "AU" or "EMRAT" is not available.
    | Fix the documentation for the fortran interface of the function calceph_prefetch.
    | Fix the return value of the function calceph_orient_unit if the frame SPICE kernel file is missing.

 * Version 2.3.1
    | Fix the compilation warnings with the Pelles compiler.
    | Fix the compilation warnings with the C89 standard.
    | Fix the compilation warnings with the GNU C compilers.
    | Fix the documentation for the constant CALCEPH_VERSION_STRING.

 * Version 2.3.0
    | Add the python interface compliant with python 2.6+ and python 3.
    | Add the preprocessor macro CALCEPH_VERSION_STRING.
    | Add the function calceph_getversion_str.
    | Add the function calceph_compute_order and calceph_orient_order.
    | Fix the return value of the functions calceph_compute_xxx when the reference frame is not available in the spice kernel files.
    | The function should produce an error  and return 0 (before the function performed no computation but it returned 1).

 * Version 2.2.5
    | Fix an incorrect result if CALCEPH_UNIT_DAY is provided to calceph_compute_unit and the target is TCG-TCB or TT-TDB.
    | Support the numerical constants declared without parenthesis in the text kernel files (.tpc).
    | Support the segment 1, 12 and 13 in the SPICE kernel file.

 * Version 2.2.4
    | Update the version number of the dynamic library.


 * Version 2.2.3
    | Add the predefined constants for calceph version in the fortran interface.
    | Fix the build chain if calceph is compiled from another folder.

 * Version 2.2.2
    | Support the compilation in the standard C89.


 * Version 2.2.1
    | Remove debug informations that are printed when errors occur in calceph\_?compute\_???.
    | Support the Portland compilers.
    | Fix the info documentation.
    | Report an error if no asteroid is available in an ephemeris file with the INPOP file format (instead of a crash).

 * Version 2.2.0
    | Support the new segments  20, 102, 103 and 120 in the SPICE kernel file.
    | Support the NAIF identification numbers.
    | Add the functions calceph_orient_unit and calceph_prefetch.

 * Version 2.1.0
    | Fix a bug in calceph_getconstant and calceph_sgetconstant with an invalid name
    | Remove the null character in the name of the constant returned by the function (f90)calceph_(s)getconstantindex when the Fortran interface is used.


 * Version 2.0.0
    | Fix memory leaks in calceph_open when errors occur.
    | Support INPOP file format 2.0 (supports TCB ephemeris file and add asteroids in the binary file).
    | Add the function calceph_open_array and calceph_compute_unit.
    | Add the tools calceph_inspector to show details about ephemeris file.
    | Support SPICE kernel file (SPK with segment 2 or 3, text and binary PCK, meta kernel, basic frame kernel).
    | Improve the performances.
    | Correct the Fortran 2003 interface for calceph_sgetconstantindex.
    | Add the constant 17 to get TCG-TCB from TCB ephemeris file.


 * Version 1.2.0
    |  Change the licensing : triple licenses to support integration in BSD software.
    |  Remove explicit dependencies on the record size for DExxx.


 * Version 1.1.2
    |  Fix a compilation warning with oracle studio compiler 12.
    |  Fix a bug with gcc on solaris in 64 bit mode.
    |  Fix the copyright statements.


 * Version 1.1.1
    |  Fix a compilation error in util.h  and a warning with the sun studio compilers.


 * Version 1.1.0
    |  Add the function calceph_seterrorhandler for the custom error handlers.


 * Version 1.0.3
    |  Support the JPL ephemeris file DE423.


 * Version 1.0.2
    | Fix memory leaks in the fortran-90 interface.

 * Version 1.0.1
    | Support the large ephemeris files (>2GB) on 32-bit operating systems.
    | Fix the documentation of the function f90calceph_sopen.
    | Fix an invalid open mode on Windows operating systems.
    | Report accurately the I/O errors.

 * Version 1.0.0
    | Initial release.


