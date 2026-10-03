/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversionjdcal.c
  \brief functions that compute jd to cal and cal to jd conversions

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
/*! Decompose a fractional day into hour, minute, and second components.

    @return 0 on success, 1 if input fd is negative

    @param fd       (in)  Fractional day (0.0–1.0)
    @param hh       (out) Pointer to integer hours [0–23]
    @param min      (out) Pointer to integer minutes [0–59]
    @param sec0     (out) Pointer to integer seconds [0–59]
    @param secfrac  (out) Pointer to fractional part of the second [0.0–1.0)
*/
/*--------------------------------------------------------------------------*/
static int decompose_fracday_to_time(double fd, int *hh, int *min, int *sec0, double *secfrac)
{
    if (fd < 0.0)
    {
        fatalerror("decompose_fracday_to_time: negative fractional day (fd < 0.0)\n");
        return 1;
    }

    fd *= 24.;
    *hh = floor(fd);
    fd -= floor(fd);
    fd *= 60.;
    *min = floor(fd);
    fd -= floor(fd);
    fd *= 60.;
    *sec0 = floor(fd);
    *secfrac = fd - floor(fd);

    return 0;
}

/*--------------------------------------------------------------------------*/
/* JD TO CALENDAR */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*! Convert a Julian Date into a Gregorian calendar date and fractional day.

    @return 0 on success, 1 if JD is out of valid range
    
    @param jd0   (in)  Integer or main part of Julian Date
    @param jdfrac (in) Fractional part of Julian Date
    @param yy    (out) Pointer to integer year (Gregorian)
    @param month (out) Pointer to integer month [1–12]
    @param day   (out) Pointer to integer day [1–31]
    @param fd    (out) Pointer to fractional day [0.0–1.0)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_to_cal_day(double jd0, double jdfrac, int *yy, int *month, int *day, double *fd)
{
    /* Minimum and maximum allowed JD */
    const double DJMIN = -68569.5;
    const double DJMAX = 1e9;

    long jd, i, l, n, k;
    double dj, f1, f2, d, s, cs, v[2], x, t, f;

    /* Verify date is acceptable. */
    dj = jd0 + jdfrac;
    if (dj < DJMIN || dj > DJMAX)
    {
        fatalerror("calceph_time_jd_to_cal_day: Julian Date out of valid range (%.10f)\n", dj);
        return 1;
    }

    /* Separate day and fraction (where -0.5 <= fraction < 0.5). */
    d = rint(jd0);
    f1 = jd0 - d;
    jd = (long) d;
    d = rint(jdfrac);
    f2 = jdfrac - d;
    jd += (long) d;

    /* Compute f1+f2+0.5 using compensated summation (Klein 2006). */
    s = 0.5;
    cs = 0.0;
    v[0] = f1;
    v[1] = f2;
    for (i = 0; i < 2; i++)
    {
        x = v[i];
        t = s + x;
        cs += fabs(s) >= fabs(x) ? (s - t) + x : (x - t) + s;
        s = t;
        if (s >= 1.0)
        {
            jd++;
            s -= 1.0;
        }
    }
    f = s + cs;
    cs = f - s;

    /* Deal with negative f. */
    if (f < 0.0)
    {
        /* Compensated summation: assume that |s| <= 1.0. */
        f = s + 1.0;
        cs += (1.0 - f) + s;
        s = f;
        f = s + cs;
        cs = f - s;
        jd--;
    }

    /* Deal with f that is 1.0 or more (when rounded to double). */
    if ((f - 1.0) >= -DBL_EPSILON / 4.0)
    {
        /* Compensated summation: assume that |s| <= 1.0. */
        t = s - 1.0;
        cs += (s - t) - 1.0;
        s = t;
        f = s + cs;
        if (-DBL_EPSILON / 2.0 < f)
        {
            jd++;
            f = f > 0. ? f : 0.;
        }
    }

    /* Express day in Gregorian calendar. */
    l = jd + 68569L;
    n = (4L * l) / 146097L;
    l -= (146097L * n + 3L) / 4L;
    i = (4000L * (l + 1L)) / 1461001L;
    l -= (1461L * i) / 4L - 31L;
    k = (80L * l) / 2447L;
    *day = (int) (l - (2447L * k) / 80L);
    l = k / 11L;
    *month = (int) (k + 2L - 12L * l);
    *yy = (int) (100L * (n - 49L) + i + l);
    *fd = f;

    /* Success. */
    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Convert a Julian Date to calendar date and time, for timescale without leap seconds.

    @return 1 on success, 0 on invalid input or conversion error

    @param jd0    (in)  Integer or main part of Julian Date
    @param jdfrac (in)  Fractional part of Julian Date
    @param yy     (out) Pointer to integer year (Gregorian)
    @param month  (out) Pointer to integer month [1–12]
    @param day    (out) Pointer to integer day [1–31]
    @param hour   (out) Pointer to integer hour [0–23]
    @param min    (out) Pointer to integer minute [0–59]
    @param sec    (out) Pointer to floating-point seconds [0.0–60.0[
*/
/*--------------------------------------------------------------------------*/
static int jd_to_cal_no_leap_timescale(double jd0, double jdfrac,
                                       int *yy, int *month, int *day, int *hour, int *min, double *sec)
{
/* GCOVR_EXCL_START */
    if (yy == NULL || month == NULL || day == NULL || hour == NULL || min == NULL || sec == NULL)
    {
        fatalerror("jd_to_cal_no_leap_timescale: null pointer argument(s)\n");
        return 0;
    }
/* GCOVR_EXCL_STOP */

    int iy1, im1, id1, js, iy2, im2, id2, hh1, min1, sec01;
    double a1, b1, fd, w, secfrac1;

    /* The two-part JD. */
    a1 = jd0;
    b1 = jdfrac;

    /* Provisional calendar date. */
    js = calceph_time_jd_to_cal_day(a1, b1, &iy1, &im1, &id1, &fd);
    if (js != 0)
        return 0;

    /* Provisional time of day. */
    sec01 = 0;
    secfrac1 = 0.;
    min1 = 0;
    hh1 = 0;
    decompose_fracday_to_time(fd, &hh1, &min1, &sec01, &secfrac1);

    /* Has the (rounded) time gone past 24h? */
    if (hh1 > 23)
    {
        js = calceph_time_jd_to_cal_day(a1 + 1.5, b1 - fd, &iy2, &im2, &id2, &w);

        if (js != 0)
            return 0;

        iy1 = iy2;
        im1 = im2;
        id1 = id2;
        hh1 = 0;
        min1 = 0;
        sec01 = 0;
    }

    /* Results. */
    *yy = iy1;
    *month = im1;
    *day = id1;
    *hour = hh1;
    *min = min1;
    *sec = (double) sec01 + secfrac1;

    /* success. */
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert a Julian Date into UTC calendar date and time, accounting for leap seconds.

    @return 0 on error, otherwise non-zero value

    @param eph    (in)  Pointer to CALCEPH binary ephemeris structure
    @param jd0    (in)  Integer or main part of Julian Date
    @param jdfrac (in)  Fractional part of Julian Date
    @param yy     (out) Pointer to integer year (Gregorian)
    @param month  (out) Pointer to integer month [1–12]
    @param day    (out) Pointer to integer day [1–31]
    @param hour   (out) Pointer to integer hour [0–23]
    @param min    (out) Pointer to integer minute [0–59]
    @param sec    (out) Pointer to floating-point seconds [0.0–61.0[
*/
/*--------------------------------------------------------------------------*/
static int jd_to_cal_utc(t_calcephbin *eph, double jd0, double jdfrac,
                         int *yy, int *month, int *day, int *hour, int *min, double *sec)
{
    int ret = 0;                /* error */

/* GCOVR_EXCL_START */
    if (eph == NULL || yy == NULL || month == NULL || day == NULL || hour == NULL || min == NULL || sec == NULL)
    {
        fatalerror("jd_to_cal_utc: null pointer argument(s)\n");
        return ret;
    }
/* GCOVR_EXCL_STOP */

    int iy1, im1, id1, js, iy2, im2, id2, hh1, min1, sec01, leap;
    double a1, b1, fd, w, secfrac1, dleap, dat0, dat12, dat24;
    long jd0_tmp;
    double jdfrac_tmp;

    /* The two-part JD. */
    a1 = jd0;
    b1 = jdfrac;

    /* Provisional calendar date. */
    js = calceph_time_jd_to_cal_day(a1, b1, &iy1, &im1, &id1, &fd);
    if (js != 0)
        return ret;

    /* Is this a leap second day? */
    leap = 0;

    /* Convert provisional YMD to JD to query delta (at 0h) */
    calceph_time_cal_to_jd_day(iy1, im1, id1, &jd0_tmp, &jdfrac_tmp);

    /* TAI-UTC at 0h today. */
    js = calceph_time_get_delta_utc_tai(eph, jd0_tmp, 0.0, &dat0);
    if (js < 0)
        return ret;

    /* TAI-UTC at 12h today (to detect drift). */
    js = calceph_time_get_delta_utc_tai(eph, jd0_tmp, 0.5, &dat12);
    if (js < 0)
        return ret;

    /* TAI-UTC at 0h tomorrow (to detect jumps). */
    js = calceph_time_jd_to_cal_day(a1 + 1.5, b1 - fd, &iy2, &im2, &id2, &w);
    if (js)
        return ret;

    /* Convert tomorrow YMD to JD */
    calceph_time_cal_to_jd_day(iy2, im2, id2, &jd0_tmp, &jdfrac_tmp);

    js = calceph_time_get_delta_utc_tai(eph, jd0_tmp, 0.0, &dat24);
    if (js != 0)
        return ret;

    /* Any sudden change in TAI-UTC (seconds). */
    dleap = dat24 - (2.0 * dat12 - dat0);

    /* If leap second day, scale the fraction of a day into SI. */
    leap = (fabs(dleap) > 0.5);
    if (leap != 0)
        fd += fd * dleap / 86400.;

    /* Provisional time of day. */
    sec01 = 0;
    secfrac1 = 0.;
    min1 = 0;
    hh1 = 0;
    decompose_fracday_to_time(fd, &hh1, &min1, &sec01, &secfrac1);

    /* Has the (rounded) time gone past 24h? */
    if (hh1 > 23)
    {
        /* Yes.  We probably need tomorrow's calendar date. */
        js = calceph_time_jd_to_cal_day(a1 + 1.5, b1 - fd, &iy2, &im2, &id2, &w);
        if (js)
            return ret;

        /* Is today a leap second day? */
        if (leap == 0)
        {
            /* No.  Use 0h tomorrow. */
            iy1 = iy2;
            im1 = im2;
            id1 = id2;
            hh1 = 0;
            min1 = 0;
            sec01 = 0;
        }
        else
        {
            /* Yes.  Are we past the leap second itself? */
            if (sec01 > 0)
            {
                /* Yes.  Use tomorrow but allow for the leap second. */
                iy1 = iy2;
                im1 = im2;
                id1 = id2;
                hh1 = 0;
                min1 = 0;
                sec01 = 0;
            }
            else
            {
                /* No.  Use 23 59 60... today. */
                hh1 = 23;
                min1 = 59;
                sec01 = 60;
            }
        }
    }

    /* Results. */
    *yy = iy1;
    *month = im1;
    *day = id1;
    *hour = hh1;
    *min = min1;
    *sec = (double) sec01 + secfrac1;

    /* sucess. */
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Dispatch Julian Date to calendar conversion according to timescale.

    Converts a two-part Julian Date into calendar date and time, using the
    appropriate timescale conversion:
      - CALCEPH_UTC → conversion with leap seconds (UTC)
      - Other timescales → continuous time (no leap seconds)

    Internally calls jd_to_cal_utc() or jd_to_cal_no_leap_timescale()
    depending on the selected timescale.

    @return 0 on error, otherwise non-zero value

    @param eph       (in)  Pointer to CALCEPH binary ephemeris (for UTC only)
    @param timescale (in)  timescale constant ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    @param jd0       (in)  Integer or main part of Julian Date
    @param jdfrac    (in)  Fractional part of Julian Date
    @param yy        (out) Pointer to integer year (Gregorian)
    @param month     (out) Pointer to integer month [1–12]
    @param day       (out) Pointer to integer day [1–31]
    @param hh        (out) Pointer to integer hour [0–23]
    @param min       (out) Pointer to integer minute [0–59]
    @param sec       (out) Pointer to floating-point seconds [0.0–61.0[
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_to_cal(t_calcephbin *eph, int timescale, double jd0, double jdfrac, int *yy, int *month, int *day,
                           int *hh, int *min, double *sec)
{
    int ret = 0;

    if (timescale == CALCEPH_UTC)
    {
        ret = jd_to_cal_utc(eph, jd0, jdfrac, yy, month, day, hh, min, sec);
    }
    else
    {
        ret = jd_to_cal_no_leap_timescale(jd0, jdfrac, yy, month, day, hh, min, sec);
    }

    return ret;
}

/*--------------------------------------------------------------------------*/
/* CALENDAR TO JD */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*! Convert a Gregorian calendar date to Julian Date day count.

    The calculation is valid for all dates later than 4800 BC.

    @return 0 on success, 1 if the date is invalid or out of range

    @param yy     (in)  Calendar year (Gregorian)
    @param month  (in)  Calendar month [1–12]
    @param day    (in)  Calendar day [1–31]
    @param djm0   (out) Pointer to MJD zero-point (always 2400000.5)
    @param djm    (out) Pointer to Modified Julian Date (MJD)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_cal_to_jd_day(int yy, int month, int day, long *jd_integer, double *jd_fraction)
{
    int j, ly, my;
    long iypmy;

    /* Earliest year allowed (4800BC) */
    const int IYMIN = -4799;

    /* Month lengths in days */
    static const int mtab[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

    /* Preset status. */
    j = 0;

    /* Validate year and month. */

    if (yy < IYMIN)
    {
        fatalerror("calceph_time_cal_to_jd_day: year (%d) below minimum allowed (%d)\n", yy, IYMIN);
        return 1;
    }

    if (month < 1 || month > 12)
    {
        fatalerror("calceph_time_cal_to_jd_day: invalid month (%d)\n", month);
        return 1;
    }

    /* If February in a leap year, 1, otherwise 0. */
    ly = ((month == 2) && !(yy % 4) && (yy % 100 || !(yy % 400)));

    /* Validate day, taking into account leap years. */
    if ((day < 1) || (day > (mtab[month - 1] + ly)))
        j = 1;

    /* Return result. */
    my = (month - 14) / 12;
    iypmy = (long) (yy + my);

    long mjd_at_noon = ((1461L * (iypmy + 4800L)) / 4L
                        + (367L * (long) (month - 2 - 12 * my)) / 12L
                        - (3L * ((iypmy + 4900L) / 100L)) / 4L + (long) day - 2432076L);

    *jd_integer = 2400000 + mjd_at_noon;
    *jd_fraction = 0.5;

    /* Return status. */
    return j;
}

/*--------------------------------------------------------------------------*/
/*! Convert a Gregorian calendar date and time to Julian Date (continuous timescale).
    No leap second correction is applied.

    @return 0 on success, 1 if the date/time is invalid

    @param eph     (in)  Pointer to CALCEPH ephemeris structure (unused)
    @param yy      (in)  Calendar year
    @param month   (in)  Calendar month [1–12]
    @param day     (in)  Calendar day [1–31]
    @param hh      (in)  Hour [0–23]
    @param min     (in)  Minute [0–59]
    @param sec     (in)  Seconds [0.0–61.0[
    @param jd0     (out) Pointer to integer part of Julian Date
    @param jdfrac  (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
static int cal_to_jd_no_leap_timescale(int yy, int month, int day,
                                       int hh, int min, double sec, double *jd0, double *jdfrac)
{
    int js;
    double nday, seclim;
    double time;
    long jd0_day;
    double jdfrac_day;

    /* Today's Julian Day Number. */
    js = calceph_time_cal_to_jd_day(yy, month, day, &jd0_day, &jdfrac_day);
    if (js != 0)
        return 1;

    /* Day length and final minute length in seconds (provisional). */
    nday = 86400;
    seclim = 60.0;

    js = 0;

    /* Validate the time. */
    if (hh >= 0 && hh <= 23)
    {
        if (min >= 0 && min <= 59)
        {
            if (sec >= 0.0)
            {
                if (sec >= seclim)
                {
                    js += 2;
                }
            }
            else
            {
                js = -6;
            }
        }
        else
        {
            js = -5;
        }
    }
    else
    {
        js = -4;
    }

    if (js < 0)
    {
        fatalerror("cal_to_jd_no_leap_timescale: invalid time (%02d:%02d:%f)\n", hh, min, sec);
        return 1;
    }

    /* The time in days (fraction of a day) */
    time = (60.0 * (60 * hh + min) + sec) / nday;

    *jd0 = jd0_day;
    *jdfrac = jdfrac_day + time;

    if (*jdfrac >= 1.0)
    {
        *jd0 += 1.0;
        *jdfrac -= 1.0;
    }

    /* Status */
    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Convert a Gregorian calendar date and UTC time to Julian Date (with leap seconds).

    @return 0 on success, 1 on invalid date/time or leap second data error

    @param eph     (in)  Pointer to CALCEPH ephemeris containing leap second data
    @param yy      (in)  Calendar year
    @param month   (in)  Calendar month [1–12]
    @param day     (in)  Calendar day [1–31]
    @param hh      (in)  Hour [0–23]
    @param min     (in)  Minute [0–59]
    @param sec     (in)  Seconds [0.0–60.0] (may include leap second)
    @param jd0     (out) Pointer to integer part of Julian Date
    @param jdfrac  (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
static int cal_to_jd_utc(t_calcephbin *eph, int yy, int month, int day, int hh, int min, double sec, double *jd0,
                         double *jdfrac)
{
    int js, iy2, im2, id2;
    long jd0_day;
    double jdfrac_day;
    double nday, seclim, dat0, dat12, dat24, dleap;
    double time;
    double fd;

    /* Today's Julian Day Number. */

    js = calceph_time_cal_to_jd_day(yy, month, day, &jd0_day, &jdfrac_day);
    if (js)
        return 1;

    /* Day length and final minute length in seconds (provisional). */
    nday = 86400.0;
    seclim = 60.0;

    /* Deal with the UTC leap second case. */

    /* TAI-UTC at 0h today. */
    js = calceph_time_get_delta_utc_tai(eph, jd0_day, 0.0, &dat0);
    if (js != 0)
        return 1;

    /* TAI-UTC at 12h today (to detect drift). */
    js = calceph_time_get_delta_utc_tai(eph, jd0_day, 0.5, &dat12);
    if (js < 0)
        return 1;

    /* TAI-UTC at 0h tomorrow (to detect jumps). */
    js = calceph_time_jd_to_cal_day(jd0_day, 1.5, &iy2, &im2, &id2, &fd);
    if (js)
        return 1;

    long jd0_tomorrow;
    double jdfrac_tomorrow;

    calceph_time_cal_to_jd_day(iy2, im2, id2, &jd0_tomorrow, &jdfrac_tomorrow);

    js = calceph_time_get_delta_utc_tai(eph, jd0_tomorrow, 0.0, &dat24);
    if (js != 0)
        return 1;

    /* Any sudden change in TAI-UTC between today and tomorrow. */
    dleap = dat24 - (2.0 * dat12 - dat0);

    /* If leap second day, correct the day and final minute lengths. */
    nday += dleap;
    if (hh == 23 && min == 59)
        seclim += dleap;

    /* End of UTC-specific actions. */
    js = 0;

    /* Validate the time. */
    if (hh >= 0 && hh <= 23)
    {
        if (min >= 0 && min <= 59)
        {
            if (sec >= 0.0)
            {
                if (sec >= seclim)
                {
                    js += 2;
                }
            }
            else
            {
                js = -6;
            }
        }
        else
        {
            js = -5;
        }
    }
    else
    {
        js = -4;
    }

    if (js < 0)
    {
        fatalerror("cal_to_jd_utc: invalid time (%02d:%02d:%f)\n", hh, min, sec);
        return 1;
    }

    /* The time in days. */
    time = (60.0 * ((60 * hh + min)) + sec) / nday;

    *jd0 = jd0_day;
    *jdfrac = jdfrac_day + time;

    if (*jdfrac >= 1.0)
    {
        *jd0 += 1.0;
        *jdfrac -= 1.0;
    }

    /* Status */
    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Dispatch calendar-to-Julian-Date conversion according to timescale.

    Converts a Gregorian date and time into a Julian Date, selecting the
    correct conversion routine based on the specified timescale:
      - CALCEPH_UTC → uses UTC conversion (with leap seconds)
      - Other timescales → uses continuous timescale conversion (no leap seconds)

    @return 0 on error, otherwise non-zero value
    @param eph       (in)  Pointer to CALCEPH ephemeris structure
    @param timescale (in)  timescale constant ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    @param yy        (in)  Calendar year
    @param month     (in)  Calendar month [1–12]
    @param day       (in)  Calendar day [1–31]
    @param hh        (in)  Hour [0–23]
    @param min       (in)  Minute [0–59]
    @param sec       (in)  Seconds [0.0–61.0[
    @param jd0       (out) Pointer to integer part of Julian Date
    @param jdfrac    (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
int calceph_time_cal_to_jd(t_calcephbin *eph, int timescale, int yy, int month, int day, int hh, int min, double sec,
                           double *jd0, double *jdfrac)
{
    int ret = 0;

    if (timescale == CALCEPH_UTC)
    {
        ret = cal_to_jd_utc(eph, yy, month, day, hh, min, sec, jd0, jdfrac);
    }
    else
    {
        ret = cal_to_jd_no_leap_timescale(yy, month, day, hh, min, sec, jd0, jdfrac);
    }

    return ret == 0 ? 1 : 0;
}

/*--------------------------------------------------------------------------*/
/* STR TO JD */
/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*! Parse and convert a time string to Julian Date.

    This function interprets a given time string in any supported format
    and converts it to a Julian Date.

    When the input timescale is 0 (undefined), the timescale inferred from
    the parsed time string is used.

    @return 1 on success, 0  on parsing or conversion failure

    @param eph       (in)  Pointer to CALCEPH ephemeris structure
    @param timescale (in)  timescale constant ( see :ref:`Timescale constant<ConstantsTimescales>` for the available timescales)
    @param str       (in)  Input time string to be parsed
    @param jd0       (out) Pointer to integer part of Julian Date
    @param jdfrac    (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
int calceph_time_str_to_jd(t_calcephbin *eph, int timescale, const char *str, double *jd0, double *jdfrac)
{
    struct calceph_time time;

    int yy, month, day, hour, min;
    double sec;

    int ret = 1;

    if (calceph_parse_time(&eph->clocale, str, &time) == 1)
        return 0;

    if (time.datatype == CALCEPH_CALENDAR)
    {
        yy = time.datetime.calendar.year;
        month = time.datetime.calendar.month;
        day = time.datetime.calendar.day;
        hour = time.datetime.calendar.hour;
        min = time.datetime.calendar.minute;
        sec = time.datetime.calendar.second;

        if (timescale == CALCEPH_TIMESCALE_FROM_STR)
            timescale = time.timescale;

        ret = calceph_time_cal_to_jd(eph, timescale, yy, month, day, hour, min, sec, jd0, jdfrac);
    }
    else
    {
        *jd0 = time.datetime.juliandate.integerpart;
        *jdfrac = time.datetime.juliandate.decimalpart;
        ret = 1;
    }

    return ret;
}
