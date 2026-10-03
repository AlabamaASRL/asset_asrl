/*-----------------------------------------------------------------*/
/*!
  \file timetesttdbspacecraftfail.c
  \brief test calceph_time_str_spacecraft_clock_to_jd_tdb and calceph_time_jd_tdb_to_str_spacecraft_clock functions

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
#include "calcephdebug.h"
#include "real.h"
#include "calceph.h"
#include "openfiles.h"

int main(void);
static void displaymsg(const char *msg);

static void displaymsg(const char *msg)
{
    printf("normal error='%s'\n", msg);
}

int main(void)
{
    int res = 0, ret;
    t_calcephbin *eph;
    t_calcephcharvalue strdate;
    double jd0_tdb, jdfrac_tdb;
    const char *kernels1[] = { "example_lsk.tls" };
    const char *kernels2[] = { "example_lsk.tls", "example_sclk.tsc" };
    calceph_seterrorhandler(3, displaymsg);

    /* missing "example_sclk.tsc" */
    eph = tests_calceph_open_array(1, kernels1);
    if (eph == NULL)
    {
        printf("failed to open kernels\n");
        return 1;
    }

    calceph_time_set_relationship_tt_tdb(eph, 1);

    ret = calceph_time_str_spacecraft_clock_to_jd_tdb(eph, 28, "1/0706865508:31865", &jd0_tdb, &jdfrac_tdb);
    if (ret != 0)
    {
        printf("calceph_time_str_spacecraft_clock_to_jd_tdb doesnot fail: %d\n", ret);
        res = 1;
    }

    ret = calceph_time_jd_tdb_to_str_spacecraft_clock(eph, 28, 2459725, 0.81458599643883644603, strdate);
    if (ret != 0)
    {
        printf("calceph_time_jd_tdb_to_str_spacecraft_clock_ doesnot fail: %d\n", ret);
        res = 1;
    }

    calceph_close(eph);

    eph = tests_calceph_open_array(2, kernels2);
    if (eph == NULL)
    {
        printf("failed to open kernels\n");
        return 1;
    }

    calceph_time_set_relationship_tt_tdb(eph, 1);

    ret = calceph_time_str_spacecraft_clock_to_jd_tdb(eph, 28, "1/x0706865508:31865", &jd0_tdb, &jdfrac_tdb);
    if (ret != 0)
    {
        printf("calceph_time_str_spacecraft_clock_to_jd_tdb doesnot fail: %d\n", ret);
        res = 1;
    }

    calceph_close(eph);

    return res;
}
