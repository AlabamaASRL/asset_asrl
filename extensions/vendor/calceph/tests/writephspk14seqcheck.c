
/*-----------------------------------------------------------------*/
/*!
  \file writephspk14seqcheck.c
  \brief check the sequential writting of a type 14 spk file

  \author  A. Durst, M. Gastineau
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

#include "writephcommoncheck.h"

int main(void)
{
    t_writephbin *w_eph;
    t_calcephbin *c_eph;
    const char segid[] = "this is string is 40 char long.........";
    double coefs[3 * (12 + 2)] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
    double coefs2[12 * (84 + 2)];
    double epochs[12];
    int j;
    double intlen = 32; /* in days */
    double radius;

    /* create a spk file */
    w_eph = writeph_spk_create("writephspk14.bsp", "writephspk14", 0);
    if (!w_eph)
    {
        printf("Error: creating the spk file failed\n");
        return 1;
    }

    /* write a comment */
    if (!writeph_comment(w_eph, "This is a comment\nwith a second line\nand a third line"))
    {
        printf("Error: writing the comment failed\n");
        return 1;
    }

    /* begin a type 14 segment */
    if (!writeph_spk14_begin(w_eph, 299, 10, 1, 2451545, 0, 2451929, 0, 1, segid))
    {
        printf("Error: beginning the type 14 segment failed\n");
        return 1;
    }

    /* prepare the epochs */
    for (j = 0; j < 12; j++)
        epochs[j] = 2451545.0 + j * intlen;

    /* add data to the type 14 segment */
    if (!writeph_spk14_add(w_eph, 3, coefs, epochs))
    {
        printf("Error: adding data to the type 14 segment failed\n");
        return 1;
    }

    /* end the type 14 segment */
    if (!writeph_spk14_end(w_eph))
    {
        printf("Error: ending the type 14 segment failed\n");
        return 1;
    }

    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file failed\n");
        return 1;
    }

    /* add a realistic type 3 segment to it */
    radius = (intlen * 86400) / 2;

    FILE *coef_file = tests_fopen("writephcoefseg3.dat", "r");
    for (j = 0; j < 12 * (84 + 2); j++)
    {
        if (j % (84 + 2) == 0)
        {       
            coefs2[j] = ((int)(j / (84 + 2)) + 0.5) * intlen * 86400;
        }
        else if (j % (84 + 2) == 1)
        {
            coefs2[j] = radius;
        }
        else if (fscanf(coef_file, "%lf", &coefs2[j]) != 1)
        {
            printf("Error: reading the coefficients failed\n");
            return 1;
        }
    }
    fclose(coef_file);

    w_eph = writeph_spk_open("writephspk14.bsp");
    if (!w_eph)
    {
        printf("Error: opening the spk file with writephfailed\n");
        return 1;
    }

    /* begin a type 14 segment */
    if (!writeph_spk14_begin(w_eph, 199, 10, 1, 2451545, 0, 2451929, 0, 13, segid))
    {
        printf("Error: beginning the type 14 segment failed\n");
        return 1;
    }

    /* add data to the type 14 segment */
    if (!writeph_spk14_add(w_eph, 12, coefs2, epochs))
    {
        printf("Error: adding data to the type 14 segment failed\n");
        return 1;
    }

    /* end the type 14 segment */
    if (!writeph_spk14_end(w_eph))
    {
        printf("Error: ending the type 14 segment failed\n");
        return 1;
    }

    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the spk file failed\n");
        return 1;
    }

    /* ---------------- checking coordinates with calceph ---------------- */

    /* open the file with the calceph module */
    c_eph = calceph_open("writephspk14.bsp");
    if (!c_eph)
    {
        printf("Error: opening the calceph file failed\n");
        return 1;
    }

    /* check the coordinates interpolated from the ephemeris file */
    if (!writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 199, 10, 1E-4, 1))
    {
        printf("Error: checking the interpolation failed\n");
        return 1;
    }

    calceph_close(c_eph);

    return 0;
}