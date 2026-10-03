/*-----------------------------------------------------------------*/
/*!
  \file writephdumprestorefail.c
  \brief check the errors of the dump and restore functions of the writeph module.
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

#include "calcephconfig.h"
#include "calceph.h"

#include "openfiles.h"

static void hidemsg(const char *msg);

int writeph_check_polynomials(t_calcephbin *eph, const char *ref_filename, int target, int center, double threshold, int is_velocity);

/*-----------------------------------------------------------------*/
/* function to hide the error message */
/*-----------------------------------------------------------------*/
static void hidemsg(const char *PARAMETER_UNUSED(msg))
{
#if HAVE_PRAGMA_UNUSED
#pragma unused(msg)
#endif
    /* printf("msg='%s'\n", msg); */
}

int main(void)
{
    /* hide the error messages */   
    calceph_seterrorhandler(3, hidemsg);

    t_writephbin *w_eph;
    t_calcephbin *c_eph;
    int j;
    const char segid[] = "this is string is 40 char long.........";

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

    /* --------------- wrong parameters --------------- */

    if (writeph_dump(NULL, "writephdumprestorefail.dump"))
    {
        printf("Error: NULL as input file does not fail\n");
        return 1;
    }

    w_eph = writeph_spk_create("writephdumprestorfail.bsp", "writephdumprestorfail", 0);
    if (!w_eph)
    {
        printf("Error: creating the spk file failed\n");
        return 1;
    }

    if (writeph_dump(w_eph, NULL))
    {
        printf("Error: NULL as output file does not fail\n");
        return 1;
    }

    /* ------------------ absent data ----------------- */

    /* add a realistic type 2 segment for the target 199 */
    if (!writeph_spk2_seq_write(w_eph, 199, 10, 1, 2451545, 0, 2451929, 0, 32, coefs, 12, 13, segid))
    {
        printf("Error: writing the type 2 segment for target 199 failed\n");
        return 1;
    }

    /* dump the file with only target 199 in it */
    if (!writeph_dump(w_eph, "writephdumprestore199.dump"))
    {
        printf("Error: dumping the file failed\n");
        return 1;
    }

    /* add a realistic type 2 segment for the target 299 */
    if (!writeph_spk2_seq_write(w_eph, 299, 10, 1, 2451545, 0, 2451929, 0, 32, coefs, 12, 13, segid))
    {
        printf("Error: writing the type 2 segment for target 299 failed\n");
        return 1;
    }

    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* open and restore the file with only target 199 */
    w_eph = writeph_spk_open("writephdumprestoreseq.bsp");
    if (!w_eph)
    {
        printf("Error: opening the spk file with writeph failed\n");
        return 1;
    }
    if (!writeph_restore(w_eph, "writephdumprestore199.dump"))
    {
        printf("Error: restoring the file failed\n");
        return 1;
    }

    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file with writeph failed\n");
        return 1;
    }

    /* open the file with the calceph module */
    c_eph = calceph_open("writephdumprestoreseq.bsp");
    if (!c_eph)
    {
        printf("Error: opening the calceph file failed\n");
        return 1;
    }

    /* try to interpolate for target 299 */
    if (writeph_check_polynomials(c_eph, "writephrefcoordinates.dat", 299, 10, 1E-10, 0))
    {
        printf("Error: checking the interpolation for target 299 does not fail.\n");
        return 1;
    }

    calceph_close(c_eph);

    return 0;
}