
/*-----------------------------------------------------------------*/
/*!
  \file writephpck3parcheck.c
  \brief check the parallel writting of a type 3 pck file

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

    /* --------------------------- declarations --------------------------- */

    t_writephbin *w_eph;
    t_calcephbin *c_eph;
    int j;

    int reserv0;
    int reserv1;
    int reserv2;

    /* data for reservation 0 */
    const int targets[] = {199, 299};
    const double intlens_jd_tdb[] = {32, 32};
    const int record_counts[] = {12, 12};
    int deg = 13;
    const char *segids[2];
    const char segid[] = "this is string is 40 char long........";
    char tmp[2][40];
    for (j = 0; j < 2; j++) {
        snprintf(tmp[j], 40, segid);
        segids[j] = tmp[j];
    }

    /* data for reservation 1 */
    int targets2[] = {399};
    double intlens_jd_tdb2[] = {32};
    int n2[] = {12};
    int deg2 = 13;
    const char *segids2[1];
    char tmp2[1][40];
    for (j = 0; j < 1; j++) {
        snprintf(tmp2[j], 40, segid);
        segids2[j] = tmp2[j];
    }

    /* data for reservation 2 */
    int targets3[] = {499};

    /* coefficients for all reservations */
    double coefs[1008];
    FILE *coef_file = tests_fopen("writephcoefseg3.dat", "r");
    for (j = 0; j < 1008; j++)
    {
        if (fscanf(coef_file, "%lf", &coefs[j]) != 1)
        {
            printf("Error: reading the coefficients failed\n");
            return 1;
        }
    }
    fclose(coef_file);

    /* ------------------------------- tests ------------------------------- */

    /* ----- sequential initialization and reservation ----- */

    /* create a pck file */
    w_eph = writeph_pck_create("writephpck3par.bsp", "writephpck3par", 0);
    if (!w_eph)
    {
        printf("Error: creating the pck file with writeph failed\n");
        return 1;
    }

    /* write a comment */
    if (!writeph_comment(w_eph, "This is a comment\nwith a second line\nand a third line"))
    {
        printf("Error: writing the comment failed\n");
        return 1;
    }

    /* reserve space for reservation 0 */
    reserv0 = writeph_pck3_par_reserve(w_eph, 2, targets, 1, 2451545, 0, 2451929, 0, intlens_jd_tdb, record_counts, deg, segids);
    if (reserv0 == 0)
    {
        printf("Error: reservation 0 failed\n");
        return 1;
    }

    /* reserve space for reservation 1 */
    reserv1 = writeph_pck3_par_reserve(w_eph, 1, targets2, 1, 2451545, 0, 2451929, 0, intlens_jd_tdb2, n2, deg2, segids2);
    if (reserv1 == 0)
    {
        printf("Error: reservation 1 failed\n");
        return 1;
    }

    /* write coefficients for target 199 */  
    if (!writeph_pck3_par_write(w_eph, reserv0, 0, 0, 12, coefs))
    {
        printf("Error: writing the coefficients failed (target 199)\n");
        return 1;
    }

    /* ------------------ parallel writing ------------------ */

    /* write coefficients for target 299 with 2 threads */
    write_two_threads(w_eph, reserv0, 1, coefs, NULL, 12, 1008, 3, 1);

    /* -------------- sequential writing -------------- */

    /* write coefficients for target 399 */
    if (!writeph_pck3_par_write(w_eph, reserv1, 0, 0, 12, coefs))
    {
        printf("Error: writing the coefficients failed (target 399)\n");
        return 1;
    }

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* -------------- writing after closing and reopening -------------- */
    w_eph = writeph_pck_open("writephpck3par.bsp");
    if (!w_eph)
    {
        printf("Error: opening the pck file with writeph failed\n");
        return 1;
    }
    /* reserve space for reservation 2 */
    reserv2 = writeph_pck3_par_reserve(w_eph, 1, targets3, 1, 2451545, 0, 2451929, 0, intlens_jd_tdb2, n2, deg2, segids2);
    if (reserv2 == 0)
    {
        printf("Error: reservation 2 failed\n");
        return 1;
    }
    /* write coefficients for target 499 */
    if (!writeph_pck3_par_write(w_eph, reserv2, 0, 0, 12, coefs))
    {
        printf("Error: writing the coefficients failed (target 499)\n");
        return 1;
    }
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* -------------- checking coordinates with calceph -------------- */

    /* open the file with the calceph module */
    c_eph = calceph_open("writephpck3par.bsp");
    if (!c_eph)
    {
        printf("Error: opening the pck file with calceph failed\n");
        return 1;
    }

    /* check the interpolated coordinates for target 199 */
    if (!writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 199, -1, 1E-10, 0))
    {
        printf("Error: checking the interpolation on target 199 failed\n");
        return 1;
    }

    /* check the interpolated coordinates for target 299 */
    if (!writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 299, -1, 1E-10, 0))
    {
        printf("Error: checking the interpolation on target 299 failed\n");
        return 1;
    }

    /* check the interpolated coordinates for target 399 */
    if (!writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 399, -1, 1E-10, 0))
    {
        printf("Error: checking the interpolation on target 399 failed\n");
        return 1;
    }

    /* check the interpolated coordinates for target 499 */
    if (!writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 499, -1, 1E-10, 0))
    {
        printf("Error: checking the interpolation on target 499 failed\n");
        return 1;
    }

    calceph_close(c_eph);

    return 0;
}