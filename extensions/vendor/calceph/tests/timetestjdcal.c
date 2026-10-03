/*-----------------------------------------------------------------*/
/*!
  \file timetestjdcal.c
  \brief test calceph_time_jd_to_cal, calceph_time_cal_to_jd and calceph_time_str_to_jd functions

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
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_MATH_H
#include <math.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif
#include "calceph.h"
#include "openfiles.h"
#include "countlines.h"

static void hidemsg(const char *msg);
int main(void);

static void hidemsg(const char *PARAMETER_UNUSED(msg))
{
#if HAVE_PRAGMA_UNUSED
#pragma unused(msg)
#endif
    /*printf("msg='%s'\n", msg) */
}

struct test
{
    t_calcephcharvalue str;
    double jd0;
    double jdfrac;
    int year;
    int month;
    int day;
    int hour;
    int minute;
    double second;
    int timescale;
    int expected_return;
};

static int test_time_jd_to_cal(struct test test)
{
    int y, mo, d, h, mi;
    double s;

    t_calcephbin *eph;

    eph = tests_calceph_open("example_lsk.tls");

    if (eph == NULL)
        return 1;

    int ret = calceph_time_jd_to_cal(eph, test.timescale, test.jd0, test.jdfrac, &y, &mo, &d, &h, &mi, &s);
    int res = 0;

    if (ret != test.expected_return)
        res = 1;

    if (ret == 1 && (test.year != y || test.month != mo || test.day != d ||
                     test.hour != h || test.minute != mi || fabs(s - test.second) > 1e-10))
    {
        res = 1;
    }

    if (res == 1)
    {
        printf("Test : %s\n", test.str);
        printf("       | CURRENT | EXPECTED\n");
        printf("  YEAR |  %-4i   |   %i\n", y, test.year);
        printf(" MONTH |   %-2i    |    %i\n", mo, test.month);
        printf("   DAY |   %-2i    |    %i\n", d, test.day);
        printf("  HOUR |   %-2i    |    %i\n", h, test.hour);
        printf("MINUTE |   %-2i    |    %i\n", mi, test.minute);
        printf("SECOND |%-.16f| %.16f\n", s, test.second);
        printf("RETURN |     %i    |   %i\n\n", ret, test.expected_return);
    }

    calceph_close(eph);

    return res;
}

static int test_time_cal_to_jd(struct test test)
{
    double jd0, jdfrac;

    t_calcephbin *eph = tests_calceph_open("example_lsk.tls");

    if (eph == NULL)
        return 1;

    int ret = calceph_time_cal_to_jd(eph, test.timescale, test.year, test.month, test.day, test.hour, test.minute,
                                     test.second, &jd0, &jdfrac);

    int res = 0;

    if (ret != test.expected_return)
        res = 1;

    double jd_total = jd0 + jdfrac;
    double jd_total_test = test.jd0 + test.jdfrac;

    if (ret == 1 && (fabs(jd_total_test - jd_total) > 1e-16))
        res = 1;

    if (res == 1)
    {
        printf("Test : %s\n", test.str);
        printf("       |     CURRENT     |     EXPECTED\n");
        printf("   JD0 | %-.16f    |  %.16f\n", jd0, test.jd0);
        printf("JDFRAC | %-.16f   |  %.16f\n", jdfrac, test.jdfrac);
        printf("RETURN |        %i        |        %i\n\n", ret, test.expected_return);
    }

    calceph_close(eph);

    return res;
}

static int test_time_str_to_jd(struct test test)
{
    double jd0, jdfrac;

    t_calcephbin *eph = tests_calceph_open("example_lsk.tls");

    if (eph == NULL)
        return 1;

    int ret = calceph_time_str_to_jd(eph, test.timescale, test.str, &jd0, &jdfrac);

    int res = 0;

    if (ret != test.expected_return)
        res = 1;

    double jd_total = jd0 + jdfrac;
    double jd_total_test = test.jd0 + test.jdfrac;

    if (ret == 1 && (fabs(jd_total_test - jd_total) > 1e-16))
        res = 1;

    if (res == 1)
    {
        printf("Test : %s\n", test.str);
        printf("       |     CURRENT     |     EXPECTED\n");
        printf("   JD0 | %.16f    |  %.16f\n", jd0, test.jd0);
        printf("JDFRAC | %.16f   |  %.16f\n", jdfrac, test.jdfrac);
        printf("RETURN |        %i        |        %i\n\n", ret, test.expected_return);
    }

    calceph_close(eph);

    return res;
}

static struct test *allocate_tests_from_file(int *nlines)
{
    FILE *file = tests_calceph_open_r("tests_cal_to_jd_utc.txt");

    int j;

    if (!file)
    {
        printf("Error when opening file");
        return NULL;
    }

    *nlines = count_lines(file);

    struct test *testarray = calloc(*nlines, sizeof(struct test));

    if (!testarray)
    {
        printf("Error when allocating array of tests");
        fclose(file);
        return NULL;
    }

    for (j = 0; j < *nlines; j++)
    {
        if (fscanf(file, "%d %d %d %d %d %lf %lf %lf",
                   &testarray[j].year,
                   &testarray[j].month,
                   &testarray[j].day,
                   &testarray[j].hour,
                   &testarray[j].minute, &testarray[j].second, &testarray[j].jd0, &testarray[j].jdfrac) != 8)
        {
            printf("Error on line %u\n", j + 1);
            break;
        }
        testarray[j].timescale = CALCEPH_UTC;
        testarray[j].expected_return = 1;
    }

    fclose(file);

    return testarray;
}

int main(void)
{
    calceph_seterrorhandler(3, hidemsg);

    int res = 0;

    int j = 0;

    int nlinesutc;

    struct test *testsutc = allocate_tests_from_file(&nlinesutc);

    if (testsutc == NULL)
    {
        return 1;
    }

    struct test tests[] = {
        {"2000-1-1T15:42:16.184 (TAI)", (double) 2451545.0,
         (double) ((3.0 * 3600.0 + 42.0 * 60.0 + 16.184) / 86400.0), 2000, 1, 1, 15, 42, 16.184, CALCEPH_TAI, 1},
        {"2016-2-17T14:50:26.2 (TT)", (double) 2457436.0,
         (double) 0.1183587962962962962962962962962963L, 2016, 2, 17, 14, 50, 26.2, CALCEPH_TT, 1},
        {"2016-2-17T14:50:26.2 (TDB)", (double) 2457436.0,
         (double) 0.1183587962962962962962962962962963L, 2016, 2, 17, 14, 50, 26.2, CALCEPH_TDB, 1},
        {"jd 2457948.76207291667 (TAI)", (double) 2457948.0L,
         (double) ((18.0L * 3600.0L + 17.0L * 60.0L + 23.1L) / 86400.0L), 2017, 7, 14, 6, 17, 23.1, CALCEPH_TAI, 1},
        {"jd 2451545. (TAI)", (double) 2451545.0L, (double) 0., 2000, 1, 1, 12, 0, 0., CALCEPH_TAI, 1},
        {"jd 2451545. (UTC)", (double) 2451545.0L, (double) 0., 2000, 1, 1, 12, 0, 0., CALCEPH_UTC, 1},
        {"jd 2451545.5 (UTC)", (double) 2451545.L, (double) 0.5, 2000, 1, 2, 0, 0, 0., CALCEPH_UTC, 1},
        {"2000 15:42:16.184", (double) -68570.0L, (double) 0.0, 0, -1, 0, 0, 0, 0, CALCEPH_TDB, 0},
        {"2029 OCT 16 13:14:11.4967158436775208", (double) 2462426.0, (double) 0.0515219527296722, 2029, 10, 16, 13, 14,
         11.4967158436775208, CALCEPH_TDB, 1},
    };

    /* General tests */
    for (j = 0; j < 9; j++)
    {
        if (test_time_jd_to_cal(tests[j]) == 1)
            res = 1;
        if (test_time_cal_to_jd(tests[j]) == 1)
            res = 1;
        if (test_time_str_to_jd(tests[j]) == 1)
            res = 1;
    }

    /* Tests on UTC Timescale */
    for (j = 0; j < nlinesutc; j++)
    {
        if (test_time_jd_to_cal(testsutc[j]) == 1)
            res = 1;
        if (test_time_cal_to_jd(testsutc[j]) == 1)
            res = 1;
    }

    free(testsutc);

    return res;
}
