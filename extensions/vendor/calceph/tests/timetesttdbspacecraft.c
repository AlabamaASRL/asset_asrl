/*-----------------------------------------------------------------*/
/*!
  \file timetesttdbspacecraft.c
  \brief test calceph_time_str_spacecraft_clock_to_jd_tdb and calceph_time_jd_tdb_to_str_spacecraft_clock functions

  \author  D. De Araujo, M. Gastineau
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
#include "calcephdebug.h"
#include "real.h"
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
   printf("msg='%s'\n", msg);
}

struct test
{
    t_calcephcharvalue spacecraft_str;
    double jd0_tdb;
    double jdfrac_tdb;
    int expected_return;
};

static int test_time_spacecraft_to_tdb(const char*filetsc, int target, struct test test)
{
    double jd0_tdb, jdfrac_tdb;

    t_calcephbin *eph;

    int ret = 0, res = 0;

    const char *kernels[] = { filetsc, "example_lsk.tls" };

    eph = tests_calceph_open_array(2, kernels);
    if (eph == NULL)
    {
        printf("failed to open kernels\n");
        return 1;
    }

    calceph_time_set_relationship_tt_tdb(eph, 1);

    ret = calceph_time_str_spacecraft_clock_to_jd_tdb(eph, target, test.spacecraft_str, &jd0_tdb, &jdfrac_tdb);

    if (ret != test.expected_return)
        res = 1;

    double cur_tdb = jd0_tdb + jdfrac_tdb;
    double ref_tdb = test.jd0_tdb + test.jdfrac_tdb;

    if (fabs(cur_tdb - ref_tdb) > 1e-9)
        res = 1;

    if (res == 1)
    {
        printf("+------------------------------------------------------------------+\n");
        printf("| Test Spacecraft -> TDB: %-40s |\n", test.spacecraft_str);
        printf("+------------+-----------------------+-----------------------------+\n");
        printf("|            |          CURRENT         |          EXPECTED        |\n");
        printf("+------------+-----------------------+-----------------------------+\n");
        printf("| jd0_tdb    | %24.16f | %24.16f |\n", jd0_tdb, test.jd0_tdb);
        printf("| jdfrac_tdb | %24.16f | %24.16f |\n", jdfrac_tdb, test.jdfrac_tdb);
        printf("+------------+--------------------------+--------------------------+\n");
        printf("| diff. TDB  |                 %16.8e                    |\n", jdfrac_tdb - test.jdfrac_tdb);
        printf("+------------+--------------------------+--------------------------+\n");
        printf("| RETURN     |             %i            |             %i            |\n", ret, test.expected_return);
        printf("+------------+--------------------------+--------------------------+\n\n");
    }

    calceph_close(eph);

    return res;
}

static int test_time_tdb_to_spacecraft(const char*filetsc, int target, struct test test)
{
    t_calcephcharvalue spacecraft_str;

    t_calcephbin *eph;

    int ret = 0, res = 0;

    const char *kernels[] = { filetsc, "example_lsk.tls" };

    eph = tests_calceph_open_array(2, kernels);
    if (eph == NULL)
    {
        printf("tests_calceph_open_array\n");
        return 1;
    }

    calceph_time_set_relationship_tt_tdb(eph, 1);

    ret = calceph_time_jd_tdb_to_str_spacecraft_clock(eph, target, test.jd0_tdb, test.jdfrac_tdb, spacecraft_str);

    if (ret != test.expected_return)
        res = 1;

    if (strcmp(spacecraft_str, test.spacecraft_str) != 0)
        res = 1;

    if (res == 1)
    {
        printf("Test TDB -> Spacecraft: jd0_tdb=%f  jdfrac_tdb=%f\n", test.jd0_tdb, test.jdfrac_tdb);
        printf("           | CURRENT | EXPECTED\n");
        printf("Spacecraft | %s | %s\n", spacecraft_str, test.spacecraft_str);
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
        if (fscanf(file, "%lf %lf %s",
                   &testarray[j].jd0_tdb, &testarray[j].jdfrac_tdb, testarray[j].spacecraft_str) != 3)
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
 
static int main_check(const char* filetsc, int target, const char* filetest)
{
    calceph_seterrorhandler(3, hidemsg);

    int res = 0;

    int j = 0;

    int nlines;

    struct test *teststdbspacecraft = allocate_tests_from_file(&nlines, filetest);

    if (teststdbspacecraft == NULL)
    {
        return 1;
    }

    /* Tests on SPACECRAFT -> TDB conversion */
    for (j = 0; j < nlines; j++)
    {
        if (test_time_spacecraft_to_tdb(filetsc, target, teststdbspacecraft[j]) == 1)
            res = 1;
    }

    /* Tests on TDB -> SPACECRAFT conversion */
    for (j = 0; j < nlines; j++)
    {
        if (test_time_tdb_to_spacecraft(filetsc, target, teststdbspacecraft[j]) == 1)
            res = 1;
    }

    free(teststdbspacecraft);

    return res;
}

int main(void)
{
    calceph_seterrorhandler(3, hidemsg);

    return main_check("example_sclk.tsc", 28, "tests_tdb_spacecraft.txt")
    +main_check("checksclk_2.tsc", 29, "tests_tdb_spacecraft2.txt"); 
}
