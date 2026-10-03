/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversiondelta.c
  \brief functions that compute deltas between different time systems

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

struct leapsecond
{
    int year, month, day;
    double leapsec;
};

/*--------------------------------------------------------------------------*/
/*! Fill the leap seconds array

   @return 0 on success, 1 on error

   @param eph           (in)    Ephemeris descriptor
 */
/*--------------------------------------------------------------------------*/
static void calceph_time_fill_leap_second(t_calcephbin *eph, struct leapsecond **pleaparray, int *nbleaparray)
{
    struct calceph_time time;
    t_calcephcharvalue *sarray;
    int ns, j, nls;

    *nbleaparray = 0;
    *pleaparray = NULL;

    /* Load leap second constants from ephemeris */
    nls = calceph_getconstantvs(eph, "DELTET/DELTA_AT", NULL, 0);
    if (nls == 0)
    {
        fatalerror("calceph_time_get_delta_utc_tai: missing DELTET/DELTA_AT string constants\n");
        return;
    }

    sarray = malloc(sizeof(t_calcephcharvalue) * nls);
    if (sarray == NULL)
    {
        fatalerror("calceph_time_get_delta_utc_tai: memory allocation failed for sarray\n");
        return;
    }

    ns = nls / 2;
    struct leapsecond *leaparray = malloc(sizeof(struct leapsecond) * ns);

    if (leaparray == NULL)
    {
        fatalerror("calceph_time_get_delta_utc_tai: memory allocation failed for leaparray\n");
        free(sarray);
        return;
    }

    calceph_getconstantvs(eph, "DELTET/DELTA_AT", sarray, nls);

    /* Parse leap second history */
    for (j = 0; j < ns; j++)
    {
        leaparray[j].leapsec = calceph_strtod(sarray[2 * j], NULL, eph->clocale);
        if (calceph_parse_time(&eph->clocale, sarray[2 * j + 1], &time) != 0 || time.datatype != CALCEPH_CALENDAR)
        {
            free(sarray);
            free(leaparray);
            return;
        }
        leaparray[j].year = time.datetime.calendar.year;
        leaparray[j].month = time.datetime.calendar.month;
        leaparray[j].day = time.datetime.calendar.day;
    }

    free(sarray);
    *nbleaparray = ns;
    *pleaparray = leaparray;
}

/*--------------------------------------------------------------------------*/
/*! Compute the difference TAI - UTC for a given UTC Julian Date.

   @return 0 on success, 1 on error

   @param eph           (in)    Ephemeris descriptor
   @param jd0_utc       (in)    Integer part of the UTC Julian date
   @param jdfrac_utc    (in)    Fractional part of the UTC Julian date
   @param delta_tai_utc (out)   Difference TAI-UTC in seconds
 */
/*--------------------------------------------------------------------------*/
int calceph_time_get_delta_utc_tai(t_calcephbin *eph, double jd0_utc, double jdfrac_utc, double *delta_tai_utc)
{
    int year, month, day;
    double fd;
    int j;
    struct leapsecond *leaparray;
    int nleapsecond;

    calceph_time_fill_leap_second(eph, &leaparray, &nleapsecond);
    if (leaparray == NULL)
    {
        return 1;
    }

    int ret = calceph_time_jd_to_cal_day(jd0_utc, jdfrac_utc, &year, &month, &day, &fd);

    if (ret != 0)
    {
        free(leaparray);
        return 1;
    }

    /* Find the applicable leap second for the given date */
    *delta_tai_utc = 0.0;
    for (j = 0; j < nleapsecond; j++)
    {
        if ((leaparray[j].year < year) ||
            (leaparray[j].year == year && leaparray[j].month < month) ||
            (leaparray[j].year == year && leaparray[j].month == month && leaparray[j].day <= day))
        {
            *delta_tai_utc = leaparray[j].leapsec;
        }
    }

    /* Handle leap second insertion point (near midnight) */
    if (fd >= 0.5)
    {
        for (j = 0; j < nleapsecond; j++)
        {
            int next_y = year, next_m = month, next_d = day + 1;

            if (next_d > 31)
            {
                next_d = 1;
                next_m++;
            }
            if (next_m > 12)
            {
                next_m = 1;
                next_y++;
            }

            if (leaparray[j].year == next_y && leaparray[j].month == next_m && leaparray[j].day == next_d)
            {
                *delta_tai_utc = leaparray[j].leapsec;
                break;
            }
        }
    }

    free(leaparray);

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Compute delta(UTC−TAI) for a given TAI Julian Date via iteration.
    This function finds the TAI-UTC offset by iteratively guessing the
    corresponding UTC date until the offset converges.

    @return 0 on success, 1 on error

    @param eph           (in)  Pointer to CALCEPH binary ephemeris structure
    @param jd0_tai       (in)  Integer or main part of Julian Date (TAI)
    @param jdfrac_tai    (in)  Fractional part of Julian Date (TAI)
    @param delta_utc_tai (out) Pointer to result variable storing delta(TAI−UTC) in seconds
*/
/*--------------------------------------------------------------------------*/
int calceph_time_get_delta_tai_utc(t_calcephbin *eph, double jd0_tai, double jdfrac_tai, double *delta_utc_tai)
{
    int j;
    int year, month, day, hour, min;
    double sec;

    struct leapsecond *leaparray;
    int nleapsecond;

    calceph_time_fill_leap_second(eph, &leaparray, &nleapsecond);

    if (leaparray == NULL)
    {
        return 1;
    }

    /* Iterative process to find UTC from TAI */
    double delta_tai_utc = 0.0;
    double prev_delta = -1.0;
    int iter = 0;
    const int max_iter = 10;
    const double convergence_threshold = 1e-9;  /* seconds */

    /* Approximate UTC JD, kept in two parts */
    double jd0_utc_approx;
    double jdfrac_utc_approx;

    while (fabs(delta_tai_utc - prev_delta) > convergence_threshold && iter < max_iter)
    {
        prev_delta = delta_tai_utc;

        /* Apply the current delta guess to the TAI fractional part */
        jd0_utc_approx = jd0_tai;
        jdfrac_utc_approx = jdfrac_tai - (delta_tai_utc / 86400.0);

        /* Normalize the two-part UTC date */
        if (jdfrac_utc_approx < 0.0)
        {
            jd0_utc_approx -= 1.0;
            jdfrac_utc_approx += 1.0;
        }
        else if (jdfrac_utc_approx >= 1.0)
        {
            jd0_utc_approx += 1.0;
            jdfrac_utc_approx -= 1.0;
        }

        /* Convert the approximate UTC JD (in two parts) to a calendar date */
        if (calceph_time_jd_to_cal(eph, CALCEPH_UTC, jd0_utc_approx, jdfrac_utc_approx,
                                   &year, &month, &day, &hour, &min, &sec) == 0)
        {
            free(leaparray);
            return 1;
        }

        /* Calculate the fractional day for the leap second lookup */
        double fd = (hour * 3600.0 + min * 60.0 + sec) / 86400.0;

        /* Find the correct TAI-UTC delta for this calendar date */
        delta_tai_utc = 0.0;
        for (j = 0; j < nleapsecond; j++)
        {
            if ((leaparray[j].year < year) ||
                (leaparray[j].year == year && leaparray[j].month < month) ||
                (leaparray[j].year == year && leaparray[j].month == month && leaparray[j].day <= day))
            {
                delta_tai_utc = leaparray[j].leapsec;
            }
        }

        /* Check for leap second applicability near midnight */
        if (fd >= 0.5)
        {
            for (j = 0; j < nleapsecond; j++)
            {
                if (leaparray[j].year == year && leaparray[j].month == month && leaparray[j].day == day + 1)
                {
                    delta_tai_utc = leaparray[j].leapsec;
                    break;
                }
            }
        }
        iter++;
    }

    free(leaparray);

    *delta_utc_tai = delta_tai_utc;

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Compute the difference TT - UTC for a given TT Julian Date.

   @return 0 on success, 1 on error

   @param eph           (in)    Ephemeris descriptor
   @param jd0_tt        (in)    Integer part of the TT Julian date
   @param jdfrac_tt     (in)    Fractional part of the TT Julian date
   @param delta_tt_utc  (out)   Difference TT-UTC in seconds
 */
/*--------------------------------------------------------------------------*/
int calceph_time_get_delta_tt_utc(t_calcephbin *eph, double jd0_tt, double jdfrac_tt, double *delta_tt_utc)
{
    double delta_tai_utc, delta_tt_tai, jd0_tai, jdfrac_tai;

    /* Get constant offset TT - TAI */
    calceph_time_get_delta_tt_tai(eph, &delta_tt_tai);

    /* Calculate TAI date from TT */
    jd0_tai = jd0_tt;
    jdfrac_tai = jdfrac_tt - delta_tt_tai / 86400.;

    if (jdfrac_tai < 0.)
    {
        jdfrac_tai += 1.0;
        jd0_tai -= 1.0;
    }

    /* Get variable offset TAI - UTC */
    if (calceph_time_get_delta_tai_utc(eph, jd0_tai, jdfrac_tai, &delta_tai_utc) != 0)
    {
        return 1;
    }

    *delta_tt_utc = delta_tt_tai + delta_tai_utc;

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Compute the difference UTC - TT for a given UTC Julian Date.

   @return 0 on success, 1 on error

   @param eph           (in)    Ephemeris descriptor
   @param jd0_utc       (in)    Integer part of the UTC Julian date
   @param jdfrac_utc    (in)    Fractional part of the UTC Julian date
   @param delta_utc_tt  (out)   Difference UTC-TT in seconds
 */
/*--------------------------------------------------------------------------*/
int calceph_time_get_delta_utc_tt(t_calcephbin *eph, double jd0_utc, double jdfrac_utc, double *delta_utc_tt)
{
    double delta_utc_tai, delta_tai_tt;

    /* Get variable offset UTC - TAI */
    if (calceph_time_get_delta_utc_tai(eph, jd0_utc, jdfrac_utc, &delta_utc_tai) != 0)
    {
        return 1;
    }

    /* Get constant offset TAI - TT */
    calceph_time_get_delta_tt_tai(eph, &delta_tai_tt);

    *delta_utc_tt = delta_utc_tai + delta_tai_tt;

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Return the constant difference TT - TAI.

   @return 0 on success

   @param eph           (in)    Ephemeris descriptor
   @param delta_tt_tai  (out)   Difference TT-TAI in seconds (nominally 32.184)
 */
/*--------------------------------------------------------------------------*/
int calceph_time_get_delta_tt_tai(t_calcephbin *eph, double *delta_tt_tai)
{
    if (eph == NULL || calceph_getconstant(eph, "DELTET/DELTA_T_A", delta_tt_tai) == 0)
    {
        *delta_tt_tai = 32.184;
    }

    return 0;
}
