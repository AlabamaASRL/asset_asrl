
/*-----------------------------------------------------------------*/
/*!
  \file writephdumprestoresparcheck.c
  \brief check the dump and restore functions of the writeph module
         for parallel writting.
         Tests are done with a type 2 segment within a spk file.

  \author  A. Durst, M. Gastineau
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

#include <stdio.h>

#include "calceph.h"
#include "openfiles.h"


int writeph_check_polynomials(t_calcephbin *eph, const char *ref_filename, int target, int center, double threshold, int is_velocity);

int main(void)
{
    t_writephbin *w_eph;
    const char segid[] = "this is string is 40 char long.........";
    int j;

    /* data for reservation 0 */
    int reserv0;
    int targets0[] = {199, 299};
    double intlens_jd_tdb0[] = {32, 32};
    int n0[] = {12, 12};
    int deg0 = 13;
    const char *segids0[2];
    char tmp[2][40];
    for (j = 0; j < 2; j++) {
        snprintf(tmp[j], 40, segid);
        segids0[j] = tmp[j];
    }

    /* data for reservation 1 */
    int reserv1;
    int targets1[] = {399};
    double intlens_jd_tdb1[] = {32};
    int n1[] = {12};
    int deg1 = 13;
    const char *segids1[1];
    char tmp1[1][40];
    for (j = 0; j < 1; j++) {
        snprintf(tmp1[j], 40, segid);
        segids1[j] = tmp1[j];
    }

    /* prepare realistic coefficients */
    double coefs[504];
    FILE *coef_file = tests_fopen("writephcoefseg2.dat", "r");
    for (j = 0; j < 504; j++)
    {
        if (fscanf(coef_file, "%lf", &coefs[j]) != 1)
        {
            printf("Error: reading the coefficients failed\n");
            return 1;
        }
    }
    fclose(coef_file);

    /* create the spk binary file */
    w_eph = writeph_spk_create("writephdumprestorepar.bsp", "writephdumprestorepar", 0);
    if (!w_eph)
    {
        printf("Error: creating the spk file failed\n");
        return 1;
    }

    /* reserve space for reservation 0 */
    reserv0 = writeph_spk2_par_reserve(w_eph, 2, targets0, 10, 1, 2451545, 0, 2451929, 0, intlens_jd_tdb0, n0, deg0, segids0);
    if (reserv0 == 0)
    {
        printf("Error: reserving space for segments failed\n");
        return 1;
    }

    if (!writeph_dump(w_eph, "writephdumprestorepar0.dump"))
    {
        printf("Error: dumping the file failed\n");
        return 1;
    }   

    /* reserve space for reservation 1 */
    reserv1 = writeph_spk2_par_reserve(w_eph, 1, targets1, 7, 1, 2451545, 0, 2451929, 0, intlens_jd_tdb1, n1, deg1, segids1);
    if (reserv1 == 0)
    {
        printf("Error: reserving space for segments failed\n");
        return 1;
    }

    if (!writeph_dump(w_eph, "writephdumprestorepar1.dump"))
    {
        printf("Error: dumping the file failed\n");
        return 1;
    }   

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    w_eph = writeph_spk_open("writephdumprestorepar.bsp");
    if (!w_eph)
    {
        printf("Error: opening the spk file with writeph failed\n");
        return 1;
    }

    if (!writeph_restore(w_eph, "writephdumprestorepar0.dump"))
    {
        printf("Error: restoring the file failed\n");
        return 1;
    }

    /* write coefficients in reservation 0 */
    if (!writeph_spk2_par_write(w_eph, reserv0, 1, 0, 12, coefs))
    {
        printf("Error: writing the coefficients failed\n");
        return 1;
    }

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    w_eph = writeph_spk_open("writephdumprestorepar.bsp");
    if (!w_eph)
    {
        printf("Error: opening the spk file with writeph failed\n");
        return 1;
    }

    if (!writeph_restore(w_eph, "writephdumprestorepar1.dump"))
    {
        printf("Error: restoring the file failed\n");
        return 1;
    }

    /* write coefficients in reservation 0 */
    if (!writeph_spk2_par_write(w_eph, reserv0, 1, 0, 12, coefs))
    {
        printf("Error: writing the coefficients failed\n");
        return 1;
    }
    
    /* write coefficients in reservation 1 */
    if (!writeph_spk2_par_write(w_eph, reserv1, 0, 0, 0, coefs))
    {
        printf("Error: writing the coefficients failed\n");
        return 1;
    }

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }
    
    return 0;
}