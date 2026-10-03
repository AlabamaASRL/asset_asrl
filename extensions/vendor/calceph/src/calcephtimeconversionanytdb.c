/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversionanytdb.c
  \brief function that compute any -> tdb conversion

  \author  D. De Araujo, M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2025-2026, CNRS
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

#if HAVE_MATH_H
/* enable M_PI with windows sdk */
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#include <float.h>
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
#include "real.h"
#include "util.h"
#include "calcephinternal.h"

/*--------------------------------------------------------------------------*/
/*! Convert a TAI Julian Date to a TDB Julian Date.
  
   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param jd0_tai    (in)      Integer part of the TAI Julian Date
   @param jdfrac_tai (in)      Fractional part of the TAI Julian Date
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
static int calceph_time_jd_tai_to_jd_tdb(t_calcephbin *eph, double jd0_tai, double jdfrac_tai, double *jd0_tdb,
                                  double *jdfrac_tdb)
{
    double delta_tt_tai;
    double jd0_tt, jdfrac_tt;

    /* Conversions: TAI -> TT -> TDB */
    calceph_time_get_delta_tt_tai(eph, &delta_tt_tai);

    jd0_tt = jd0_tai;
    jdfrac_tt = jdfrac_tai + delta_tt_tai / 86400.0;

    /* Normalize */
    if (jdfrac_tt >= 1.0)
    {
        jd0_tt += 1.0;
        jdfrac_tt -= 1.0;
    }

    return calceph_time_jd_tt_to_jd_tdb(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a UTC Julian Date to a TDB Julian Date.
  
   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param jd0_utc    (in)      Integer part of the UTC Julian Date
   @param jdfrac_utc (in)      Fractional part of the UTC Julian Date
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
static int calceph_time_jd_utc_to_jd_tdb(t_calcephbin *eph, double jd0_utc, double jdfrac_utc, double *jd0_tdb,
                                  double *jdfrac_tdb)
{
    double delta_utc_tt;
    double jd0_tt, jdfrac_tt;

    /* Conversions: UTC -> TT -> TDB */
    if (calceph_time_get_delta_utc_tt(eph, jd0_utc, jdfrac_utc, &delta_utc_tt) != 0)
    {
        return 0;
    }

    jd0_tt = jd0_utc;
    jdfrac_tt = jdfrac_utc + delta_utc_tt / 86400.0;

    /* Normalize */
    if (jdfrac_tt >= 1.0)
    {
        jd0_tt += 1.0;
        jdfrac_tt -= 1.0;
    }

    return calceph_time_jd_tt_to_jd_tdb(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a time string to a Julian Date in the TDB timescale.
  
   This function parses the input string to determine the date and its 
   original timescale (UTC, TAI, TT, TDB), then performs the necessary 
   conversions using the ephemeris data to produce a TDB Julian Date.

   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param str        (in)      Date/time string to parse
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int calceph_time_str_any_to_jd_tdb(t_calcephbin *eph, const char *str, double *jd0_tdb, double *jdfrac_tdb)
{
    int ret = 0;

/* GCOVR_EXCL_START */
    if (eph == NULL)
    {
        fatalerror("calceph_time_str_any_to_jd_tdb: null eph pointer\n");
        return ret;
    }

    if (str == NULL || jd0_tdb == NULL || jdfrac_tdb == NULL)
    {
        fatalerror("calceph_time_str_any_to_jd_tdb: null pointer argument\n");
        return ret;
    }
/* GCOVR_EXCL_STOP */

    struct calceph_time time;
    double jd0_in, jdfrac_in;

    /* Parse the string */
    if (calceph_parse_time(&eph->clocale, str, &time) != 0)
    {
        return ret;
    }

    if (calceph_time_str_to_jd(eph, time.timescale, str, &jd0_in, &jdfrac_in) == 0)
    {
        return ret;
    }

    /* Convert the input JD (jd0_in, jdfrac_in) from its native timescale to TDB. */
    switch (time.timescale)
    {
        case CALCEPH_TDB:
            /* No conversion needed */
            *jd0_tdb = jd0_in;
            *jdfrac_tdb = jdfrac_in;
            ret = 1;
            break;

        case CALCEPH_TT:
            /* Conversion: TT -> TDB */
            ret = calceph_time_jd_tt_to_jd_tdb(eph, jd0_in, jdfrac_in, jd0_tdb, jdfrac_tdb);
            break;

        case CALCEPH_TAI:
            ret = calceph_time_jd_tai_to_jd_tdb(eph, jd0_in, jdfrac_in, jd0_tdb, jdfrac_tdb);
            break;

        case CALCEPH_UTC:
            ret = calceph_time_jd_utc_to_jd_tdb(eph, jd0_in, jdfrac_in, jd0_tdb, jdfrac_tdb);
            break;

        case CALCEPH_TCB:
            /* Conversion: TCB -> TDB */
            ret = calceph_time_jd_tcb_to_jd_tdb(eph, jd0_in, jdfrac_in, jd0_tdb, jdfrac_tdb);
            break;

        default:
            fatalerror("calceph_time_str_any_to_jd_tdb: unsupported or undefined input time scale (%d)\n",
                       time.timescale);
            break;
    }

    return ret;
}
