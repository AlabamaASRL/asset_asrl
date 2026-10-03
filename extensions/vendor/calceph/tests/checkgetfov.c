/*-----------------------------------------------------------------*/
/*!
  \file checkgetfov.c
  \brief Check that the results of calceph_getfov are correct.

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

#include <stdio.h>
#include "calceph.h"
#include "openfiles.h"
#include "calcephconfig.h"
#if HAVE_MATH_H
#include <math.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif

int main(void);

static void print_values_on_error(int instrumentid, double *expected_bounds, double *current_bounds,
                                  double expected_vector[3], double current_vector[3], int nvalues, int ret, int shape,
                                  t_calcephcharvalue frame)
{
    int j;

    printf("find invalid value for instument %i:\n", instrumentid);
    printf("+---------------+----------------+----------------+\n");
    printf("|    vector     |    expected    |     current    |\n");
    printf("+---------------+----------------+----------------+\n");
    for (j = 0; j < nvalues; j++)
    {
        printf("| v_bounds%-2i    | %14.9f | %14.9f |\n", j / 3, expected_bounds[j], current_bounds[j]);
    }
    printf("+---------------+----------------+----------------+\n");
    for (j = 0; j < 3; j++)
    {
        printf("| v_boresight%-2i | %14.9f | %14.9f |\n", j / 3, expected_vector[j], current_vector[j]);
    }
    printf("+---------------+----------------+----------------+\n");
    printf("|                 ret = %-2i                        |\n", ret);
    printf("|               shape = %-i                         |\n", shape);
    printf("|               frame = %-20s      |\n", frame);
    printf("+-------------------------------------------------+\n");
}

static int test_fov(t_calcephbin *peph, int instrumentid, double *expected_bounds,
                    int expected_ret, int expected_shape, char *expected_frame, double expected_vector[3])
{
    double vector[3];

    double arraybounds[256];

    int shape;

    t_calcephcharvalue frame;

    int ret;

    int res = 0;

    int j;

    int nbounds = calceph_getfov(peph, instrumentid, &shape, frame, vector, NULL, 0);

    if (nbounds <= 0)
    {
        return 1;
    }

    ret = calceph_getfov(peph, instrumentid, &shape, frame, vector, arraybounds, nbounds);

    if (ret != expected_ret || shape != expected_shape || strcmp(frame, expected_frame) != 0)
    {
        res = 1;
    }

    for (j = 0; res == 0 && j < 3 * nbounds; j++)
    {
        if (fabs(arraybounds[j] - expected_bounds[j]) > 1e-6)
        {
            res = 1;
        }
    }

    for (j = 0; res == 0 && j < 3; j++)
    {
        if (fabs(vector[j] - expected_vector[j]) > 1e-9)
        {
            res = 1;
        }
    }

    if (res != 0)
    {
        print_values_on_error(instrumentid, expected_bounds, arraybounds, expected_vector, vector, 3 * nbounds, ret,
                              shape, frame);
    }

    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    t_calcephbin *peph;
    int res = 0;
    double expected_vector[3];

    /* open file ik */
    peph = tests_calceph_open("example_ik.ti");

    /* TESTS WITH ANGLES */
    /* test instrument -42550 (circle) */
    double expected_circle[3] = { -3.046734, 0.000000, 24.813654 };
    expected_vector[0] = 0.0;
    expected_vector[1] = 0.0;
    expected_vector[2] = 25.0;

    if (test_fov(peph, -42550, expected_circle, 1, 3, "EXAMPLE_CIRCLE", expected_vector) != 0)
        res = 1;

    /* test instrument -42554 (ellipse) */
    double expected_ellipse[6] = { 0.063601, 0.997022, 0.043619,
        0.171929, 0.985109, 0.000000
    };
    expected_vector[0] = 0.0636614381316129;
    expected_vector[1] = 0.997971553349531;
    expected_vector[2] = 0.0;

    if (test_fov(peph, -42554, expected_ellipse, 2, 4, "EXAMPLE_ELLIPSE_2", expected_vector) != 0)
        res = 1;

    /* test instrument -42552 (rectangle) */
    double expected_rectangle[12] = { 0.063657, 0.000004, 0.997972,
        0.063657, -0.000004, 0.997972,
        0.063666, -0.000004, 0.997971,
        0.063666, 0.000004, 0.997971
    };
    expected_vector[0] = 0.0636614381316129;
    expected_vector[1] = 0.0;
    expected_vector[2] = 0.997971553349531;

    if (test_fov(peph, -42552, expected_rectangle, 4, 2, "EXAMPLE_RECTANGLE", expected_vector) != 0)
        res = 1;

    /* TESTS WITH BOUNDARY CORNERS */
    /* test instrument -42551 (ellipse) */
    double expected_ellipse2[6] = { 1.0, 0.0, 0.01745506,
        1.0, 0.03492077, 0.0
    };
    expected_vector[0] = 1.0;
    expected_vector[1] = 0.0;
    expected_vector[2] = 0.0;

    if (test_fov(peph, -42551, expected_ellipse2, 2, 4, "EXAMPLE_ELLIPSE", expected_vector) != 0)
        res = 1;

    /* test instrument -42553 (polygon) */
    double expected_polygon[9] = { 0.0, 0.8, 0.5,
        0.4, 0.8, -0.2,
        -0.4, 0.8, -0.2
    };
    expected_vector[0] = 0.0;
    expected_vector[1] = 1.0;
    expected_vector[2] = 0.0;

    if (test_fov(peph, -42553, expected_polygon, 3, 1, "EXAMPLE_POLYGON", expected_vector) != 0)
        res = 1;

    /* close file ik */
    calceph_close(peph);
    return res;
}
