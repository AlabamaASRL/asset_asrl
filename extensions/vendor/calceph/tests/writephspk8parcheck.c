
/*-----------------------------------------------------------------*/
/*!
  \file writephspk8parcheck.c
  \brief check the parallel writting of a type 8 spk file

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
    const double intlens_jd_tdb[] = {0.09, 0.09};
    const int record_counts[] = {4056, 4056};
    int deg = 9;
    const char *segids[2];
    const char segid[] = "this is string is 40 char long........";
    char tmp[2][40];
    for (j = 0; j < 2; j++) {
        snprintf(tmp[j], 40, segid);
        segids[j] = tmp[j];
    }

    /* data for reservation 1 */
    int targets2[] = {399};
    double intlens_jd_tdb2[] = {0.09};
    int N2[] = {4056};
    int deg2 = 9;
    const char *segids2[1];
    char tmp2[1][40];
    for (j = 0; j < 1; j++) {
        snprintf(tmp2[j], 40, segid);
        segids2[j] = tmp2[j];
    }

    /* data for reservation 2 */
    int targets3[] = {499};

    /* states for all reservations */
    double states[4056 * 6];
    double dates[4056];
    FILE *states_file = tests_fopen("writephrefcoordinates.dat", "r");
    for (j = 0; j < 4056; j++)
    {
        if (fscanf(states_file, "%lf %lf %lf %lf %lf %lf %lf", &dates[j], &states[j*6], &states[j*6+1], &states[j*6+2], &states[j*6+3], &states[j*6+4], &states[j*6+5]) != 7)
        {
            printf("Error: reading the states failed (line %d)\n", j);
            return 1;
        }
    }
    fclose(states_file);

    double julian_start = dates[0];
    double julian_stop = dates[4055];
    double start_jd0 = (int)julian_start;
    double start_frac = julian_start - start_jd0;
    double end_jd0 = (int)julian_stop;
    double end_frac = julian_stop - end_jd0;

    /* ------------------------------- tests ------------------------------- */

    /* ----- sequential initialization and reservation ----- */

    /* create a spk file */
    w_eph = writeph_spk_create("writephspk8par.bsp", "writephspk8par", 0);
    if (!w_eph)
    {
        printf("Error: creating the spk file with writeph failed\n");
        return 1;
    }

    /* write a comment */
    if (!writeph_comment(w_eph, "This is a comment\nwith a second line\nand a third line"))
    {
        printf("Error: writing the comment failed\n");
        return 1;
    }

    /* reserve space for reservation 0 */
    reserv0 = writeph_spk8_par_reserve(w_eph, 2, targets, 10, 1, start_jd0, start_frac, end_jd0, end_frac, intlens_jd_tdb, record_counts, deg, segids);
    if (reserv0 == 0)
    {
        printf("Error: reservation 0 failed\n");
        return 1;
    }

    /* reserve space for reservation 1 */
    reserv1 = writeph_spk8_par_reserve(w_eph, 1, targets2, 7, 1, start_jd0, start_frac, end_jd0, end_frac, intlens_jd_tdb2, N2, deg2, segids2);
    if (reserv1 == 0)
    {
        printf("Error: reservation 1 failed\n");
        return 1;
    }

    /* write states for target 199 */  
    if (!writeph_spk8_par_write(w_eph, reserv0, 0, 0, 4056, states))
    {
        printf("Error: writing the states failed (target 199)\n");
        return 1;
    }

    /* ------------------ parallel writing ------------------ */

    /* write states for target 299 with 2 threads */
    write_two_threads(w_eph, reserv0, 1, states, NULL, 4056, 6 * 4056, 8, 0);

    /* -------------- sequential writing -------------- */

    /* write states for target 399 */
    if (!writeph_spk8_par_write(w_eph, reserv1, 0, 0, 4056, states))
    {
        printf("Error: writing the states failed (target 399)\n");
        return 1;
    }

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* -------------- writing after closing and reopening -------------- */
    w_eph = writeph_spk_open("writephspk8par.bsp");
    if (!w_eph)
    {
        printf("Error: opening the file with writeph failed\n");
        return 1;
    }
    /* reserve space for reservation 2 */
    reserv2 = writeph_spk8_par_reserve(w_eph, 1, targets3, 5, 1, start_jd0, start_frac, end_jd0, end_frac, intlens_jd_tdb2, N2, deg2, segids2);
    if (reserv2 == 0)
    {
        printf("Error: reservation 2 failed\n");
        return 1;
    }
    /* write states for target 499 */
    if (!writeph_spk8_par_write(w_eph, reserv2, 0, 0, 4056, states))
    {
        printf("Error when writing the states\n");
        return 1;
    }
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* -------------- checking coordinates with calceph -------------- */

    /* open the file with the calceph module */
    c_eph = calceph_open("writephspk8par.bsp");
    if (!c_eph)
    {
        printf("Error: opening the file with calceph failed\n");
        return 1;
    }

    /* check the interpolated coordinates for target 199 */
    if (!writeph_check_states(c_eph, "writephrefcoordinateslagrange.dat", 199, 10, 1E-7))
    {
        printf("Error: checking the interpolation failed (target 199)\n");
        return 1;
    }

    /* check the interpolated coordinates for target 299 */
    if (!writeph_check_states(c_eph, "writephrefcoordinateslagrange.dat", 299, 10, 1E-7))
    {
        printf("Error: checking the interpolation failed (target 299)\n");
        return 1;
    }

    /* check the interpolated coordinates for target 399 */
    if (!writeph_check_states(c_eph, "writephrefcoordinateslagrange.dat", 399, 7, 1E-7))
    {
        printf("Error: checking the interpolation failed (target 399)\n");
        return 1;
    }

    /* check the interpolated coordinates for target 499 */
    if (!writeph_check_states(c_eph, "writephrefcoordinateslagrange.dat", 499, 5, 1E-7))
    {
        printf("Error: checking the interpolation failed (target 499)\n");
        return 1;
    }

    calceph_close(c_eph);

    return 0;
}