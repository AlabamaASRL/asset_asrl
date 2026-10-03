/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversionutctdb.c
  \brief functions that compute utc <-> tdb conversions

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
/*! Convert a UTC time string to a Julian Date in the TDB timescale.
 
   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param str        (in)      UTC time string to parse
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int calceph_time_str_utc_to_jd_tdb(t_calcephbin *eph, const char *str, double *jd0_tdb, double *jdfrac_tdb)
{
/* GCOVR_EXCL_START */
    if (eph == NULL)
    {
        fatalerror("calceph_time_str_utc_to_jd_tdb: null eph pointer\n");
        return 0;
    }

    if (jd0_tdb == NULL || jdfrac_tdb == NULL)
    {
        fatalerror("calceph_time_str_utc_to_jd_tdb: null pointer argument (jd0_tdb or jdfrac_tdb)\n");
        return 0;
    }
/* GCOVR_EXCL_STOP */

    double jd0_utc, jdfrac_utc, sec_utc;
    int year_utc, month_utc, day_utc, hour_utc, min_utc;

    /* Parse string to UTC Julian Date */
    if (calceph_time_str_to_jd(eph, CALCEPH_UTC, str, &jd0_utc, &jdfrac_utc) == 0)
        return 0;

    /* Convert UTC JD to UTC Calendar components */
    if (calceph_time_jd_to_cal
        (eph, CALCEPH_UTC, jd0_utc, jdfrac_utc, &year_utc, &month_utc, &day_utc, &hour_utc, &min_utc, &sec_utc) == 0)
        return 0;

    return calceph_time_cal_utc_to_jd_tdb(eph, year_utc, month_utc, day_utc, hour_utc, min_utc, sec_utc, jd0_tdb,
                                          jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a UTC Calendar date to a Julian Date in the TDB timescale.

   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param yy         (in)      Year
   @param month      (in)      Month
   @param day        (in)      Day
   @param hh         (in)      Hour
   @param min        (in)      Minute
   @param sec        (in)      Second
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int calceph_time_cal_utc_to_jd_tdb(t_calcephbin *eph, int yy, int month, int day, int hh, int min, double sec,
                                   double *jd0_tdb, double *jdfrac_tdb)
{
    if (eph == NULL)
    {
        fatalerror("calceph_time_cal_utc_to_jd_tdb: null eph pointer\n");
        return 0;
    }

    if (jd0_tdb == NULL || jdfrac_tdb == NULL)
    {
        fatalerror("calceph_time_cal_utc_to_jd_tdb: null pointer argument (jd0_tdb or jdfrac_tdb)\n");
        return 0;
    }

    double jd0_tt, jdfrac_tt, delta_utc_tt;

    /* Convert Calendar UTC to JD UTC (as intermediate step for TT) */
    if (calceph_time_cal_to_jd(eph, CALCEPH_UTC, yy, month, day, hh, min, sec, &jd0_tt, &jdfrac_tt) == 0)
        return 0;

    /* Get offset UTC -> TT */
    if (calceph_time_get_delta_utc_tt(eph, jd0_tt, jdfrac_tt, &delta_utc_tt) != 0)
        return 0;

    /* Apply offset to get TT */
    jdfrac_tt += delta_utc_tt / 86400.;

    if (jdfrac_tt >= 1.0)
    {
        jd0_tt += 1.;
        jdfrac_tt -= 1.0;
    }

    /* Convert TT to TDB */
    return calceph_time_jd_tt_to_jd_tdb(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a TDB Julian Date to a UTC Calendar date.

   @return 0 on error, otherwise non-zero value

   @param eph        (in)      Ephemeris descriptor
   @param jd0_tdb    (in)      Integer part of the TDB Julian Date
   @param jdfrac_tdb (in)      Fractional part of the TDB Julian Date
   @param yy         (out)     Year
   @param month      (out)     Month
   @param day        (out)     Day
   @param hh         (out)     Hour
   @param min        (out)     Minute
   @param sec        (out)     Second
 */
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tdb_to_cal_utc(t_calcephbin *eph, double jd0_tdb, double jdfrac_tdb, int *yy, int *month, int *day,
                                   int *hh, int *min, double *sec)
{
    if (eph == NULL)
    {
        fatalerror("calceph_time_jd_tdb_to_cal_utc: null pointer output argument\n");
        return 0;
    }

    if (yy == NULL || month == NULL || day == NULL || hh == NULL || min == NULL || sec == NULL)
    {
        fatalerror("calceph_time_jd_tdb_to_cal_utc: null pointer output argument\n");
        return 0;
    }

    double jd0_tt, jdfrac_tt, delta_tt_utc;

    /* Convert TDB to TT */
    if (calceph_time_jd_tdb_to_jd_tt(eph, jd0_tdb, jdfrac_tdb, &jd0_tt, &jdfrac_tt) == 0)
        return 0;

    /* Get offset TT -> UTC */
    if (calceph_time_get_delta_tt_utc(eph, jd0_tt, jdfrac_tt, &delta_tt_utc) != 0)
        return 0;

    /* Apply offset to get UTC Julian Date */
    double jd0_utc = jd0_tt;
    double jdfrac_utc = jdfrac_tt - (delta_tt_utc / 86400.0);

    if (jdfrac_utc < 0.0)
    {
        jd0_utc -= 1.0;
        jdfrac_utc += 1.0;
    }
    else if (jdfrac_utc >= 1.0)
    {
        jd0_utc += 1.0;
        jdfrac_utc -= 1.0;
    }

    double delta_check;

    if (calceph_time_get_delta_utc_tt(eph, jd0_utc, jdfrac_utc, &delta_check) != 0)
        return 0;

    if (delta_check != delta_tt_utc)
    {
        delta_tt_utc = delta_check;
        jd0_utc = jd0_tt;
        jdfrac_utc = jdfrac_tt - (delta_tt_utc / 86400.0);

        if (jdfrac_utc < 0.0)
        {
            jd0_utc -= 1.0;
            jdfrac_utc += 1.0;
        }
        else if (jdfrac_utc >= 1.0)
        {
            jd0_utc += 1.0;
            jdfrac_utc -= 1.0;
        }
    }

    /* Convert UTC JD to Calendar components */
    return calceph_time_jd_to_cal(eph, CALCEPH_UTC, jd0_utc, jdfrac_utc, yy, month, day, hh, min, sec);
}
