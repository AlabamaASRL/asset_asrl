/*-----------------------------------------------------------------*/
/*!
  \file timetestanytdb.c
  \brief test calceph_time_str_any_to_jd_tdb function

  \author  D. De Araujo, M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2025, CNRS
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
#define __CALCEPH_WITHIN_CALCEPH 1
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

struct test_any_tdb
{
    t_calcephcharvalue str_in;
    double jd0_tdb_expected;
    double jdfrac_tdb_expected;
    int expected_return;
};

static int test_time_any_to_tdb(struct test_any_tdb test)
{
    double jd0_tdb_current, jdfrac_tdb_current;
    t_calcephbin *eph;

    eph = tests_calceph_open("example_lsk.tls");
    if (eph == NULL)
        return 1;

    calceph_time_set_relationship_tt_tdb(eph, 1);

    int ret = calceph_time_str_any_to_jd_tdb(eph, test.str_in, &jd0_tdb_current, &jdfrac_tdb_current);

    int res = 0;

    if (ret != test.expected_return)
        res = 1;

    double cur_tdb = jd0_tdb_current + jdfrac_tdb_current;
    double ref_tdb = test.jd0_tdb_expected + test.jdfrac_tdb_expected;

    if (fabs(cur_tdb - ref_tdb) > 1e-9)
        res = 1;

    if (res == 1)
    {
        printf("Test STR ANY -> TDB JD : INPUT \"%s\"\n", test.str_in);
        printf("             |        CURRENT         |        EXPECTED\n");
        printf("JD TDB (SUM) |%-24.16f|%.16f\n", cur_tdb, ref_tdb);
        printf("RETURN       |           %i            |           %i\n\n", ret, test.expected_return);
    }

    calceph_close(eph);

    return res;
}

static struct test_any_tdb *allocate_tests_from_file(int *nlines, const char *filename)
{
    FILE *file = tests_calceph_open_r(filename);
    int j;

    if (!file)
    {
        printf("Error when opening file: %s\n", filename);
        return NULL;
    }

    *nlines = count_lines(file);
    if (*nlines == 0)
    {
        printf("No lines found in test file: %s\n", filename);
        fclose(file);
        return NULL;
    }

    struct test_any_tdb *testarray = calloc(*nlines, sizeof(struct test_any_tdb));

    if (!testarray)
    {
        printf("Error when allocating array of tests\n");
        fclose(file);
        return NULL;
    }

    char line_buffer[256];

    for (j = 0; j < *nlines; j++)
    {
        if (fgets(line_buffer, sizeof(line_buffer), file) == NULL)
        {
            printf("Error reading line %u\n", j + 1);
            free(testarray);
            fclose(file);
            return NULL;
        }

        char *semicolon1 = strchr(line_buffer, ';');

        if (semicolon1 == NULL)
        {
            printf("Error parsing line %u: No first semicolon found.\n", j + 1);
            free(testarray);
            fclose(file);
            return NULL;
        }

        char *semicolon2 = strchr(semicolon1 + 1, ';');

        if (semicolon2 == NULL)
        {
            printf("Error parsing line %u: No second semicolon found.\n", j + 1);
            free(testarray);
            fclose(file);
            return NULL;
        }

        *semicolon1 = '\0';

        strncpy(testarray[j].str_in, line_buffer, sizeof(testarray[j].str_in) - 1);
        testarray[j].str_in[sizeof(testarray[j].str_in) - 1] = '\0';

        int len = strlen(testarray[j].str_in);

        while (len > 0)
        {
            char c = testarray[j].str_in[len - 1];

            if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            {
                len--;
            }
            else
            {
                break;
            }
        }
        testarray[j].str_in[len] = '\0';

        char *start = testarray[j].str_in;

        while (*start == ' ' || *start == '\t')
        {
            start++;
        }
        if (start != testarray[j].str_in)
        {
            memmove(testarray[j].str_in, start, strlen(start) + 1);
        }

        testarray[j].jd0_tdb_expected = strtod(semicolon1 + 1, NULL);

        testarray[j].jdfrac_tdb_expected = strtod(semicolon2 + 1, NULL);

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
    int nlinesanytdb;

    struct test_any_tdb *testsanytdb = allocate_tests_from_file(&nlinesanytdb, "tests_any_tdb.txt");

    if (testsanytdb == NULL)
    {
        printf("Fail to load file tests_any_tdb.txt\n");
        return 1;
    }

    for (j = 0; j < nlinesanytdb; j++)
    {
        if (test_time_any_to_tdb(testsanytdb[j]) == 1)
        {
            res = 1;
        }
    }

    free(testsanytdb);

    return res;
}
