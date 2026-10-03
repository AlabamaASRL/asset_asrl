/*-----------------------------------------------------------------*/
/*!
  \file f2003calcephtime.c
  \brief Fortran 2003 interface for the time functions.

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright, 2025-2026,CNRS
   email of the author : Mickael.Gastineau@obspm.fr

*/
/*-----------------------------------------------------------------*/

/*-----------------------------------------------------------------*/
/* License  of this file :
 This file is "triple-licensed", you have to choose one  of the three licenses
 below to apply on this file.

    CeCILL-C
        The CeCILL-C license is close to the GNU LGPL.
        ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )

 or CeCILL-B
        The CeCILL-B license is close to the BSD.
        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)

 or CeCILL v2.1
      The CeCILL license is compatible with the GNU GPL.
      ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )

This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under
French law and abiding by the rules of distribution of free software.
You can  use, modify and/ or redistribute the software under the terms
of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA
at the following URL "http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its
terms.
*/
/*-----------------------------------------------------------------*/

#include "calcephconfig.h"
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif

#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"

#if __STRICT_ANSI__
#define inline
#endif

#include "f90fillspace.h"

int f2003calceph_time_jd_tdb_to_str_spacecraft_clock(t_calcephbin *eph, int target, double jd0_tdb, double jdfrac_tdb,
                                                     t_calcephcharvalue str);
                                                     
/*--------------------------------------------------------------------------*/
/*! Convert TDB (Julian Date) to Spacecraft Clock String.

    Computes the spacecraft clock string corresponding to a given Barycentric
    Dynamical Time (TDB). The function performs the reverse operation of the
    SCLK parsing:
    1. Converts TDB to the parallel time system (e.g., TT) if required.
    2. Finds the appropriate rate coefficient record.
    3. Computes the elapsed time and converts it to clock ticks.
    4. Identifies the correct partition.
    5. Formats the ticks into fields (RIM:MOD) and produces the output string.

    @return 0 on error, otherwise non-zero value

    @param eph          (in)  ephemeris object
    @param target       (in)  NAIF ID of the spacecraft
    @param jd0_tdb      (in)  Integer part of Julian Date (TDB)
    @param jdfrac_tdb   (in)  Fractional part of Julian Date (TDB)
    @param str          (out) Output string buffer (must be large enough, e.g., 64 chars)
*/
/*--------------------------------------------------------------------------*/
int f2003calceph_time_jd_tdb_to_str_spacecraft_clock(t_calcephbin *eph, int target, double jd0_tdb, double jdfrac_tdb,
                                                     t_calcephcharvalue str)
{
    str[0] = '\0';
    int ret = calceph_time_jd_tdb_to_str_spacecraft_clock(eph, target, jd0_tdb, jdfrac_tdb,
                                                          str);

    if (ret == 1)
        calceph_fortranfillspace(str, CALCEPH_MAX_CONSTANTVALUE);
    return ret;
}
