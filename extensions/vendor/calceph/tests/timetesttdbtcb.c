/*-----------------------------------------------------------------*/
/*!
  \file timetesttdbtcb.c
  \brief test calceph_time_jd_tdb_to_jd_tcb and calceph_time_jd_tcb_to_jd_tdb functions

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2026, CNRS
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
    double jd0_tcb;
    double jdfrac_tcb;
    double jd0_tdb;
    double jdfrac_tdb;
    int expected_return;
};

static int test_time_tdb_tcb(struct test test)
{
    double jd0_tcb, jdfrac_tcb, jd0_tdb, jdfrac_tdb;

    t_calcephbin *eph;

    eph = tests_calceph_open("example_lsk.tls");
    if (eph == NULL)
        return 1;

    /* TEST TCB->TDB */
    int ret = calceph_time_jd_tcb_to_jd_tdb(eph, test.jd0_tcb, test.jdfrac_tcb, &jd0_tdb, &jdfrac_tdb);
    int res = 0;

    if (ret != test.expected_return)
        res = 1;

    double cur_tdb = jd0_tdb + jdfrac_tdb;
    double ref_tdb = test.jd0_tdb + test.jdfrac_tdb;

    if (fabs(cur_tdb - ref_tdb) > 1e-9)
        res = 1;

    if (res == 1)
    {
        printf("Test TCB->TDB : JD %.20f\n", test.jd0_tcb + test.jdfrac_tcb);
        printf("           | CURRENT | EXPECTED\n");
        printf("jd0_tdb    |%.12f|%.12f\n", jd0_tdb, test.jd0_tdb);
        printf("jdfrac_tdb |%.12f|%.12f\n", jdfrac_tdb, test.jdfrac_tdb);
        printf("RETURN     |     %i    |   %i\n\n", ret, test.expected_return);
    }

    /* TEST TDB->TCB */
    res = 0;

    ret = calceph_time_jd_tdb_to_jd_tcb(eph, test.jd0_tdb, test.jdfrac_tdb, &jd0_tcb, &jdfrac_tcb);

    if (ret != test.expected_return)
        res = 1;

    double cur_tcb = jd0_tcb + jdfrac_tcb;
    double ref_tcb = test.jd0_tcb + test.jdfrac_tcb;

    if (fabs(cur_tcb - ref_tcb) > 1e-9)
        res = 1;

    if (res == 1)
    {
        printf("Test TDB->TCB : JD %.20f\n", test.jd0_tdb + test.jdfrac_tdb);
        printf("           | CURRENT | EXPECTED\n");
        printf("jd0_tcb     |%.12f|%.12f\n", jd0_tcb, test.jd0_tcb);
        printf("jdfrac_tcb  |%.12f|%.12f\n", jdfrac_tcb, test.jdfrac_tcb);
        printf("RETURN     |     %i    |   %i\n\n", ret, test.expected_return);
    }

    calceph_close(eph);

    return res;
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
        if (fscanf(file, "%lf %lf %lf %lf",
                   &testarray[j].jd0_tdb, &testarray[j].jdfrac_tdb, &testarray[j].jd0_tcb, &testarray[j].jdfrac_tcb) != 4)
        {
            printf("Error on line %u\n", j + 1);
            free(testarray);
            fclose(file);
            return NULL;
        }
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

    int nlinestdbtcb;

    struct test *teststdbtcb = allocate_tests_from_file(&nlinestdbtcb, "tests_tdb_tcb.txt");

    if (teststdbtcb == NULL)
    {
        return 1;
    }

    for (j = 0; j < nlinestdbtcb; j++)
    {
        if (test_time_tdb_tcb(teststdbtcb[j]) == 1)
            res = 1;
    }

    free(teststdbtcb);

    return res;
}
