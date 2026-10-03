/*-----------------------------------------------------------------*/
/*!
  \file timetestutctdb.c
  \brief test calceph_time_cal_utc_to_jd_tdb and calceph_time_str_utc_to_jd_tdb functions

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
#include "util.h"
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
    t_calcephcharvalue str_utc;
    int yy_utc;
    int month_utc;
    int day_utc;
    int hour_utc;
    int min_utc;
    double sec_utc;
    double jd0_tdb;
    double jdfrac_tdb;
    int expected_return;
};

static int test_time_utc_tdb(struct test test)
{
    double jd0_tdb, jdfrac_tdb, sec_utc;
    int yy_utc, month_utc, day_utc, hour_utc, min_utc;

    t_calcephbin *eph;

    eph = tests_calceph_open("example_lsk.tls");
    if (eph == NULL)
        return 1;

    calceph_time_set_relationship_tt_tdb(eph, 1);

    int yy, month, day, hour, min;
    double sec;

    calceph_time_jd_to_cal(eph, CALCEPH_TDB, test.jd0_tdb, test.jdfrac_tdb, &yy, &month, &day, &hour, &min, &sec);

    /* TEST UTC CAL -> TDB JD */
    int ret =
        calceph_time_cal_utc_to_jd_tdb(eph, test.yy_utc, test.month_utc, test.day_utc, test.hour_utc, test.min_utc,
                                       test.sec_utc, &jd0_tdb, &jdfrac_tdb);
    int res = 0, total_res = 0;

    if (ret != test.expected_return)
        res = 1;

    double cur_tdb = jd0_tdb + jdfrac_tdb;
    double ref_tdb = test.jd0_tdb + test.jdfrac_tdb;

    if (fabs(cur_tdb - ref_tdb) > 1e-12)
        res = 1;

    if (res == 1)
    {
        printf("Test UTC Calendar -> TDB JD : CALENDAR %i-%i-%iT%i:%i:%f\n", test.yy_utc, test.month_utc, test.day_utc,
               test.hour_utc, test.min_utc, test.sec_utc);
        printf("           |        CURRENT         |        EXPECTED\n");
        printf("jd0_tdb    |%-24.16f|%.16f\n", jd0_tdb, test.jd0_tdb);
        printf("jdfrac_tdb |%-24.16f|%.16f\n", jdfrac_tdb, test.jdfrac_tdb);
        printf("RETURN     |           %i            |           %i\n\n", ret, test.expected_return);
    }

    if (res != 0)
        total_res = 1;

    /* TEST UTC STR -> TDB JD */
    ret = calceph_time_str_utc_to_jd_tdb(eph, test.str_utc, &jd0_tdb, &jdfrac_tdb);
    res = 0;

    if (ret != test.expected_return)
        res = 1;

    cur_tdb = jd0_tdb + jdfrac_tdb;
    ref_tdb = test.jd0_tdb + test.jdfrac_tdb;

    if (fabs(cur_tdb - ref_tdb) > 1e-12)
        res = 1;

    if (res == 1)
    {
        printf("Test UTC str -> TDB JD : CALENDAR %i-%i-%iT%i:%i:%f\n", test.yy_utc, test.month_utc, test.day_utc,
               test.hour_utc, test.min_utc, test.sec_utc);
        printf("           |        CURRENT         |        EXPECTED\n");
        printf("jd0_tdb    |%-24.16f|%.16f\n", jd0_tdb, test.jd0_tdb);
        printf("jdfrac_tdb |%-24.16f|%.16f\n", jdfrac_tdb, test.jdfrac_tdb);
        printf("RETURN     |           %i            |           %i\n\n", ret, test.expected_return);
    }

    if (res != 0)
        total_res = 1;

    /* TEST TDB JD -> UTC CAL */
    ret =
        calceph_time_jd_tdb_to_cal_utc(eph, test.jd0_tdb, test.jdfrac_tdb, &yy_utc, &month_utc, &day_utc, &hour_utc,
                                       &min_utc, &sec_utc);
    res = 0;

    if (ret != test.expected_return)
        res = 1;

    if (test.yy_utc != yy_utc || test.month_utc != month_utc || test.day_utc != day_utc || test.hour_utc != hour_utc ||
        test.min_utc != min_utc || fabs(sec_utc - test.sec_utc) > 1e-6)
        res = 1;

    if (res == 1)
    {
        printf("Test TDB JD -> UTC Calendar : JD %.16f\n", test.jd0_tdb + test.jdfrac_tdb);
        printf("           |   CURRENT   |   EXPECTED\n");
        printf("YEAR       |    %-5i    |    %i\n", yy_utc, test.yy_utc);
        printf("MONTH      |    %-2i       |     %i\n", month_utc, test.month_utc);
        printf("DAY        |    %-2i       |     %i\n", day_utc, test.day_utc);
        printf("HOUR       |    %-2i       |     %i\n", hour_utc, test.hour_utc);
        printf("MINUTE     |    %-2i       |     %i\n", min_utc, test.min_utc);
        printf("SECOND     |%-.9f | %-.9f\n", sec_utc, test.sec_utc);
        printf("RETURN     |      %i      |      %i\n\n", ret, test.expected_return);
    }

    if (res != 0)
        total_res = 1;

    calceph_close(eph);

    return total_res;
}

static struct test *allocate_tests_from_file(int *nlines, const char *filename)
{
    FILE *file = tests_calceph_open_r(filename);

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
        if (fscanf(file, "%i %i %i %i %i %lf %lf %lf",
                   &testarray[j].yy_utc, &testarray[j].month_utc, &testarray[j].day_utc, &testarray[j].hour_utc,
                   &testarray[j].min_utc, &testarray[j].sec_utc, &testarray[j].jd0_tdb, &testarray[j].jdfrac_tdb) != 8)
        {
            printf("Error on line %u\n", j + 1);
            free(testarray);
            fclose(file);
            return NULL;
        }
        testarray[j].expected_return = 1;
        calceph_snprintf( testarray[j].str_utc, CALCEPH_MAX_CONSTANTVALUE, "%i-%i-%iT%i:%i:%.16f", testarray[j].yy_utc,
                testarray[j].month_utc, testarray[j].day_utc, testarray[j].hour_utc,
                testarray[j].min_utc, testarray[j].sec_utc);
    }
    fclose(file);

    return testarray;
}

static int test_on_day_with_leapseconds(void)
{
    int yy, mo, dd, hh, mi;
    int ret1 = 0, ret2 = 0, ret = 0;
    double ss, jdi10, jdf10, jd0_TDB;
    int sec;
    t_calcephbin *peph = tests_calceph_open("example_lsk.tls");

    /* UTC : 1995-12-31T23:59:59.5E0 */

    calceph_time_set_relationship_tt_tdb(peph, 1);

    ret1 = calceph_time_cal_utc_to_jd_tdb(peph, 1995, 12, 31, 23, 59, 59.5E0, &jdi10, &jdf10);
    ret2 = calceph_time_jd_tdb_to_cal_utc(peph, jdi10, jdf10, &yy, &mo, &dd, &hh, &mi, &ss);
    if (ret1 != 1 || ret2 != 1 || yy != 1995 || mo != 12 || dd != 31 || hh != 23 ||
        mi != 59 || fabs(ss - 59.5) > 1e-4 || (jdi10 + jdf10) - 2450083.5007023601792753 > 1e-8)
    {
        printf("test_on_day_with_leapseconds: (1995-12-31T23:59:59.5)\n");
        printf("expected TDB=2450083 0.5007023601792753\n");
        printf("computed TDB=%23.16E %23.16E\n", jdi10, jdf10);
        printf("difference  TDB (seconds)=%f \n", (jdi10 + jdf10) - 2450083.5007023601792753);
        printf("expected date UTC = 1995-12-31T23:59:59.50\n");
        printf("computed date UTC = %d-%d-%dT%d:%d:%.5f\n\n", yy, mo, dd, hh, mi, ss);
        ret = 1;
    }

    /* UTC : 1995-12-31T23:59:60.5E0 */

    ret1 = calceph_time_cal_utc_to_jd_tdb(peph, 1995, 12, 31, 23, 59, 60.5E0, &jdi10, &jdf10);
    ret2 = calceph_time_jd_tdb_to_cal_utc(peph, jdi10, jdf10, &yy, &mo, &dd, &hh, &mi, &ss);
    if (ret1 != 1 || ret2 != 1 || yy != 1995 || mo != 12 || dd != 31 || hh != 23 ||
        mi != 59 || fabs(ss - 60.5) > 1e-4 || (jdi10 + jdf10) - 2450083.5007139341905713 > 1e-8)
    {
        printf("test_on_day_with_leapseconds: (1995-12-31T23:59:60.5)\n");
        printf("expected TDB=2450083 0.5007139341905713\n");
        printf("computed TDB=%23.16E %23.16E\n", jdi10, jdf10);
        printf("difference  TDB (seconds)=%f \n", (jdi10 + jdf10) - 2450083.5007139341905713);
        printf("expected date UTC = 1995-12-31T23:59:60.50\n");
        printf("computed date UTC = %d-%d-%dT%d:%d:%.5f\n\n", yy, mo, dd, hh, mi, ss);
        ret = 1;
    }

    /* UTC : 1996-1-1T0:0:0.50 */

    ret1 = calceph_time_cal_utc_to_jd_tdb(peph, 1996, 1, 1, 0, 0, 0.5, &jdi10, &jdf10);
    ret2 = calceph_time_jd_tdb_to_cal_utc(peph, jdi10, jdf10, &yy, &mo, &dd, &hh, &mi, &ss);
    if (ret1 != 1 || ret2 != 1 || yy != 1996 || mo != 1 || dd != 1 || hh != 0 ||
        mi != 0 || fabs(ss - 0.5) > 1e-4 || (jdi10 + jdf10) - 2450083.5007255082018673 > 1e-8)
    {
        printf("test_on_day_with_leapseconds: (1996-1-1T0:0:0.50)\n");
        printf("expected TDB=2450083 0.5007255082018673\n");
        printf("computed TDB=%23.16E %23.16E\n", jdi10, jdf10);
        printf("difference  TDB (seconds)=%f \n", (jdi10 + jdf10) - 2450083.5007255082018673);
        printf("expected date UTC = 1996-1-1T0:0:0.50\n");
        printf("computed date UTC = %d-%d-%dT%d:%d:%.5f\n\n", yy, mo, dd, hh, mi, ss);
        ret = 1;
    }

    /* UTC : 1995-12-31T23:59:[0,59].5 */

    jd0_TDB = 2450083.500019498;

    for (sec = 0; sec < 60; sec++)
    {
        ret1 = calceph_time_cal_utc_to_jd_tdb(peph, 1995, 12, 31, 23, 59, sec + 0.5E0, &jdi10, &jdf10);
        ret2 = calceph_time_jd_tdb_to_cal_utc(peph, jdi10, jdf10, &yy, &mo, &dd, &hh, &mi, &ss);
        if (ret1 != 1 || ret2 != 1 || yy != 1995 || mo != 12 || dd != 31 || hh != 23 ||
            mi != 59 || fabs(ss - (sec + 0.5)) > 1e-4 ||
            fabs(((jdi10 + jdf10) - (jd0_TDB + sec / 86400.)) * 86400.) > 1e-3)
        {
            printf("expected TDB=%.16f\n", (jd0_TDB + sec / 86400.));
            printf("computed TDB=%.16f\n", (jdi10 + jdf10));
            printf("difference  TDB (seconds)=%f \n", ((jdi10 + jdf10) - (jd0_TDB + sec / 86400.)) * 86400);
            printf("expected date UTC = %d %d %d %d %d %.5f\n", 1995, 12, 31, 23, 59, sec + 0.5E0);
            printf("computed date UTC = %d %d %d %d %d %.5f\n", yy, mo, dd, hh, mi, ss);
            printf("\n");

            ret = 1;
        }
    }

    /* UTC : 1996-1-1T00:00:[0,59].5 */

    jd0_TDB = 2450083.5000194897875190 + 60 / 86400.;

    for (sec = 0; sec < 60; sec++)
    {
        ret1 = calceph_time_cal_utc_to_jd_tdb(peph, 1996, 1, 1, 0, 0, sec + 0.5E0, &jdi10, &jdf10);
        ret2 = calceph_time_jd_tdb_to_cal_utc(peph, jdi10, jdf10, &yy, &mo, &dd, &hh, &mi, &ss);
        if (ret1 != 1 || ret2 != 1 || yy != 1996 || mo != 1 || dd != 1 || hh != 0 ||
            mi != 0 || fabs(ss - (sec + 0.5)) > 1e-4)
        {
            printf("expected TDB=%.16f\n", (jd0_TDB + sec / 86400.));
            printf("computed TDB=%.16f\n", (jdi10 + jdf10));
            printf("difference  TDB (seconds)=%f \n", ((jdi10 + jdf10) - (jd0_TDB + sec / 86400.)) * 86400);
            printf("expected date UTC = %d %d %d %d %d %.5f\n", 1996, 1, 1, 0, 0, sec + 0.5E0);
            printf("computed date UTC = %d %d %d %d %d %.5f\n", yy, mo, dd, hh, mi, ss);
            printf("\n");

            ret = 1;
        }
    }

    if (peph != NULL)
        calceph_close(peph);

    return ret;
}

int main(void)
{
    calceph_seterrorhandler(3, hidemsg);

    int res = 0;

    int j = 0;

    int nlinesutctdb;

    struct test *testsutctdb = allocate_tests_from_file(&nlinesutctdb, "tests_utc_tdb.txt");

    if (testsutctdb == NULL)
    {
        return 1;
    }

    /* Tests on TT <-> TDB conversion with model 0 */
    for (j = 0; j < nlinesutctdb; j++)
    {
        if (test_time_utc_tdb(testsutctdb[j]) == 1)
            res = 1;
    }

    free(testsutctdb);

    /* TESTS on a day with leap seconds */

    if (test_on_day_with_leapseconds() == 1)
        res = 1;

    return res;
}
