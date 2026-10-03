/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversiontdbtt.c
  \brief functions that compute tt <-> tdb conversions

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
/*! Perform a compensated summation (two-sum) of two doubles.
    Computes sum = a + b and err = (a + b) - sum, but robustly.
    (Based on Knuth/Dekker's algorithm)

    @param a    (in)  First number
    @param b    (in)  Second number
    @param sum  (out) The floating point sum (a + b)
    @param err  (out) The rounding error from the summation
*/
/*--------------------------------------------------------------------------*/
static void compensated_two_sum(double a, double b, double *sum, double *err)
{
    *sum = a + b;
    double b_virtual = *sum - a;
    double a_virtual = *sum - b_virtual;

    *err = (a - a_virtual) + (b - b_virtual);
}

/*--------------------------------------------------------------------------*/
/*! set the relationship between TT and TDB time scales

    Two models are supported:
      - model = 0 : use of calceph_compute_unit()
      - model = 1 : use the default model based on the TLS file

    The function updates the internal field `relationship_tdb_tt` in the
    ephemeris structure.

    @return 0 on error, otherwise non-zero value

    @param eph   (inout) ephemeris object
    @param model (in)    relationship model (0 or 1)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_set_relationship_tt_tdb(t_calcephbin *eph, int model)
{
    if (eph == NULL || (model != 0 && model != 1))
    {
        fatalerror("calceph_time_set_relationship_tt_tdb: invalid input (eph=%p, model=%d)\n", eph, model);
        return 0;
    }

    eph->relationship_tdb_tt = model;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TT using model 1

    The conversion applies the following steps:
      - Retrieve constants DELTET/K, DELTET/EB, DELTET/M
      - Compute mean anomaly M = M0 + M1 * t
      - Compute eccentric anomaly E = M + EB * sin(M)
      - Compute delta = K * sin(E)
      - Apply correction jd_tt = jd_tdb - delta / 86400

    @return 0 on error, otherwise non-zero value

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tt     (out) integer part of Julian Date (TT)
    @param jdfrac_tt  (out) fractional part of Julian Date (TT)
*/
/*--------------------------------------------------------------------------*/

static int jd_tdb_to_jd_tt_model1(t_calcephbin *eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tt,
                                  double *jdfrac_tt)
{
    double K, EB, Marray[2], M, E, delta;
    double t_large, t_small, t_sum, t_err;
    double m_large, m_small, M_sum, M_err;

    if (calceph_getconstant(eph, "DELTET/K", &K) != 1 ||
        calceph_getconstant(eph, "DELTET/EB", &EB) != 1 || calceph_getconstantvd(eph, "DELTET/M", Marray, 2) != 2)
    {
        fatalerror("'DELTET/K', 'DELTET/EB' or 'DELTET/M' is missing");
        return 0;
    }

    /* Compute t = ( (jd0_tdb - 2451545.0) + jdfrac_tdb ) * 86400.0 */
    /* Use compensated sum to avoid precision loss */
    t_large = (jd0_tdb - 2451545.0) * 86400.0;
    t_small = jdfrac_tdb * 86400.0;
    compensated_two_sum(t_large, t_small, &t_sum, &t_err);

    /* Compute M = M0 + M1 * t, preserving precision */
    /* M = M0 + M1 * (t_sum + t_err) = (M0 + M1*t_sum) + (M1*t_err) */
    m_large = Marray[0] + Marray[1] * t_sum;
    m_small = Marray[1] * t_err;
    compensated_two_sum(m_large, m_small, &M_sum, &M_err);

    /* M is now a high-precision sum; combine for sin() */
    M = M_sum + M_err;
    E = M + EB * sin(M);

    delta = K * sin(E);

    /* Compute output JD: (jd0_tdb + jdfrac_tdb) - (delta / 86400.0) */
    /* We split this into integer and fractional parts robustly */

    double jd0 = jd0_tdb;
    double delta_days = delta / 86400.0;
    double jdfrac_corr, jdfrac_err;

    /* Compute high-precision fractional part: jdfrac_tdb - delta_days */
    compensated_two_sum(jdfrac_tdb, -delta_days, &jdfrac_corr, &jdfrac_err);

    double jdfrac = jdfrac_corr + jdfrac_err;

    /* Normalize the fractional part */
    if (jdfrac < 0.0)
    {
        jdfrac += 1.0;
        jd0 -= 1.0;
    }
    else if (jdfrac >= 1.0)
    {
        jdfrac -= 1.0;
        jd0 += 1.0;
    }

    *jd0_tt = jd0;
    *jdfrac_tt = jdfrac;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TT using model 0

    @return 0 on error, otherwise non-zero value

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tt     (out) integer part of Julian Date (TT)
    @param jdfrac_tt  (out) fractional part of Julian Date (TT)
*/
/*--------------------------------------------------------------------------*/
static int jd_tdb_to_jd_tt_model0(t_calcephbin *eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tt,
                                  double *jdfrac_tt)
{
    double PV[6], delta, jd_tt;
    int ret;

    ret = calceph_compute_unit(eph, jd0_tdb, jdfrac_tdb, NAIFID_TIME_TTMTDB, NAIFID_TIME_CENTER,
                               CALCEPH_USE_NAIFID + CALCEPH_UNIT_SEC, PV);

    if (ret == 0)
    {
        return 0;
    }

    delta = PV[0];

    jd_tt = (jd0_tdb + jdfrac_tdb) + delta / 86400.0;

    *jd0_tt = floor(jd_tt);
    *jdfrac_tt = jd_tt - *jd0_tt;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TT according to selected model

    @return 0 on error, otherwise non-zero value

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tt     (out) integer part of Julian Date (TT)
    @param jdfrac_tt  (out) fractional part of Julian Date (TT)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tdb_to_jd_tt(t_calcephbin *eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tt,
                                 double *jdfrac_tt)
{
/* GCOVR_EXCL_START */
    if (eph == NULL || jd0_tt == NULL || jdfrac_tt == NULL)
    {
        fatalerror("calceph_time_jd_tdb_to_jd_tt: invalid input (eph=%p, jd0_tt=%p, jdfrac_tt=%p)\n", eph, jd0_tt,
                   jdfrac_tt);
        return 0;
    }
/* GCOVR_EXCL_STOP */

    int ret;

    if (eph->relationship_tdb_tt == 0)
    {
        ret = jd_tdb_to_jd_tt_model0(eph, jd0_tdb, jdfrac_tdb, jd0_tt, jdfrac_tt);
    }
    else if (eph->relationship_tdb_tt == 1)
    {
        ret = jd_tdb_to_jd_tt_model1(eph, jd0_tdb, jdfrac_tdb, jd0_tt, jdfrac_tt);
    }
    else
    {
        fatalerror("calceph_time_jd_tdb_to_jd_tt: invalid relationship_tdb_tt=%d\n", eph->relationship_tdb_tt);
        ret = 0;
    }

    return ret;
}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TT to TDB using model 0 (via iteration)

    @return 0 on error, otherwise non-zero value

    @param eph         (in)  ephemeris object
    @param jd0_tt      (in)  integer part of Julian Date (TT)
    @param jdfrac_tt   (in)  fractional part of Julian Date (TT)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
static int jd_tt_to_jd_tdb_model0(t_calcephbin *eph, double jd0_tt, double jdfrac_tt, double *jd0_tdb,
                                  double *jdfrac_tdb)
{
    const int maxLoop = 10;
    const double epsilon = DBL_EPSILON;
    int loopCount = 0, ret;
    double PV[6];
    double delta;
    double jd0_tdb_approx = jd0_tt; /* Initial guess for TDB = TT */
    double jdfrac_tdb_approx = jdfrac_tt;
    double jd_tdb_sum_old, jd_tdb_sum_new;

    do
    {
        /* Store the combined sum only for the convergence check */
        jd_tdb_sum_old = jd0_tdb_approx + jdfrac_tdb_approx;

        /* Get the (TT-TDB) delta using our current TDB guess */
        ret = calceph_compute_unit(eph,
                                   jd0_tdb_approx, jdfrac_tdb_approx,
                                   NAIFID_TIME_TTMTDB, NAIFID_TIME_CENTER, CALCEPH_USE_NAIFID + CALCEPH_UNIT_SEC, PV);

        if (ret == 0)
            return 0;

        /* PV[0] = delta = TT - TDB (in seconds) */
        delta = PV[0];

        /* Calculate new TDB guess: TDB = TT - delta */
        jd0_tdb_approx = jd0_tt;
        jdfrac_tdb_approx = jdfrac_tt - (delta / 86400.0);

        /* Normalize the new TDB two-part date */
        if (jdfrac_tdb_approx < 0.0)
        {
            jd0_tdb_approx -= 1.0;
            jdfrac_tdb_approx += 1.0;
        }
        else if (jdfrac_tdb_approx >= 1.0)
        {
            jd0_tdb_approx += 1.0;
            jdfrac_tdb_approx -= 1.0;
        }

        /* Get the new sum for the convergence check */
        jd_tdb_sum_new = jd0_tdb_approx + jdfrac_tdb_approx;

        loopCount++;
    }
    while (fabs(jd_tdb_sum_new - jd_tdb_sum_old) > epsilon && loopCount <= maxLoop);

    if (loopCount > maxLoop)
    {
        fatalerror("jd_tt_to_jd_tdb_model0: no convergence after %d iterations\n", maxLoop);
        return 0;
    }

    *jd0_tdb = jd0_tdb_approx;
    *jdfrac_tdb = jdfrac_tdb_approx;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TT to TDB using model 1

    The model uses an analytical expression involving the constants
    DELTET/K, DELTET/EB, and DELTET/M to compute the TT–TDB difference.

    @return 0 on error, otherwise non-zero value

    @param eph         (in)  ephemeris object
    @param jd0_tt      (in)  integer part of Julian Date (TT)
    @param jdfrac_tt   (in)  fractional part of Julian Date (TT)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
static int jd_tt_to_jd_tdb_model1(t_calcephbin *eph, double jd0_tt, double jdfrac_tt, double *jd0_tdb,
                                  double *jdfrac_tdb)
{
    double K, EB, Marray[2], M, E, delta, jd_tdb;
    double t_large, t_small, t_sum, t_err, JD2000;
    double m_large, m_small, M_sum, M_err;

    int retK, retEB, retM;

    retK = calceph_getconstant(eph, "DELTET/K", &K);
    retEB = calceph_getconstant(eph, "DELTET/EB", &EB);
    retM = calceph_getconstantvd(eph, "DELTET/M", Marray, 2);

    if (retK != 1 || retEB != 1 || retM != 2)
    {
        fatalerror("jd_tt_to_jd_tdb_model1: missing DELTET constants\n");
        return 0;
    }

    JD2000 = 2451545.0;

    /* Compute t = ( (jd0_tt - JD2000) + jdfrac_tt ) * 86400.0 */
    /* Use compensated sum to avoid precision loss */
    t_large = (jd0_tt - JD2000) * 86400.0;
    t_small = jdfrac_tt * 86400.0;
    compensated_two_sum(t_large, t_small, &t_sum, &t_err);

    /* Compute M = M0 + M1 * t, preserving precision */
    /* M = M0 + M1 * (t_sum + t_err) = (M0 + M1*t_sum) + (M1*t_err) */
    m_large = Marray[0] + Marray[1] * t_sum;
    m_small = Marray[1] * t_err;
    compensated_two_sum(m_large, m_small, &M_sum, &M_err);

    /* M is now a high-precision sum; combine for sin() */
    M = M_sum + M_err;
    E = M + EB * sin(M);

    delta = K * sin(E);

    /* Compute output JD: (jd0_tt + jdfrac_tt) + (delta / 86400.0) */
    /* We sum all three parts with high precision */
    double delta_days = delta / 86400.0;
    double sum1, err1, sum2, err2;

    /* (jd0_tt + jdfrac_tt) */
    compensated_two_sum(jd0_tt, jdfrac_tt, &sum1, &err1);

    /* (sum1) + delta_days */
    compensated_two_sum(sum1, delta_days, &sum2, &err2);

    /* The final high-precision sum is (sum2 + err1 + err2) */
    jd_tdb = sum2 + err1 + err2;

    *jd0_tdb = floor(jd_tdb);
    *jdfrac_tdb = jd_tdb - *jd0_tdb;

    return 1;

}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TT to TDB according to the selected model
    defined in the ephemeris object.

    Depending on the value of eph->relationship_tdb_tt, either model 0 or model 1 is applied.

    @return 0 on error, otherwise non-zero value

    @param eph         (in)  ephemeris object
    @param jd0_tt      (in)  integer part of Julian Date (TT)
    @param jdfrac_tt   (in)  fractional part of Julian Date (TT)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tt_to_jd_tdb(t_calcephbin *eph, double jd0_tt, double jdfrac_tt, double *jd0_tdb,
                                 double *jdfrac_tdb)
{
/* GCOVR_EXCL_START */
    if (eph == NULL || jd0_tdb == NULL || jdfrac_tdb == NULL)
    {
        fatalerror("calceph_time_jd_tt_to_jd_tdb: invalid input (eph=%p, jd0_tdb=%p, jdfrac_tdb=%p)\n", eph, jd0_tdb,
                   jdfrac_tdb);
        return 1;
    }
/* GCOVR_EXCL_STOP */

    int ret = 0;

    if (eph->relationship_tdb_tt == 0)
    {
        ret = jd_tt_to_jd_tdb_model0(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb);
    }
    else if (eph->relationship_tdb_tt == 1)
    {
        ret = jd_tt_to_jd_tdb_model1(eph, jd0_tt, jdfrac_tt, jd0_tdb, jdfrac_tdb);
    }
    else
    {
        fatalerror
            ("calceph_time_jd_tt_to_jd_tdb: relationship_tdb_tt is incorrect in eph, and should be initialized with calceph_time_set_relationship_tt_tdb (value=%i)\n",
             eph->relationship_tdb_tt);
        ret = 0;
    }

    return ret;
}
