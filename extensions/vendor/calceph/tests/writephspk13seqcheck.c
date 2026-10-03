
/*-----------------------------------------------------------------*/
/*!
  \file writephspk13seqcheck.c
  \brief check the sequential writting of a type 13 spk file

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
    double states1[] = {1, 2, 3, 4, 5, 6, 7, 8, 13, 10, 11, 12};
    double epochs1[] = {2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545, 2451545};
    int j;

    /* create a spk file */
    w_eph = writeph_spk_create("writephspk13.bsp", "writephspk13", 0);
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

    /* add some type 13 segments to it */
    for (j = 0; j < 85; j++)
    {
        if (!writeph_spk13_seq_write(w_eph, 299, 5, 1, 2451545, 0, 2451929, 0, states1, epochs1, 2, 9, segid))
        {
            printf("Error: writing the type 13 segment (index %d) failed\n", j);
            return 1;
        }
    }
    
    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file failed\n");
        return 1;
    }

    /* add a realistic type 13 segment to it */
    double lines[4056 * 7];
    double states2[4056 * 6];
    double epochs2[4056];
    FILE *states_file = tests_fopen("writephrefcoordinates.dat", "r");
    for (j = 0; j < 4056; j++)
    {
        if (fscanf(states_file, "%lf %lf %lf %lf %lf %lf %lf", &lines[j*7], &lines[j*7+1], &lines[j*7+2], &lines[j*7+3], &lines[j*7+4], &lines[j*7+5], &lines[j*7+6]) != 7)
        {
                printf("Error: reading the states failed (line %d)\n", j);
                return 1;
        }

        epochs2[j] = lines[j*7];
        memcpy(&states2[j*6], &lines[j*7 + 1], 6 * sizeof(double));
    }
    fclose(states_file);

    double julian_start = lines[0];
    double julian_stop = lines[4055 * 7];
    double start_jd0 = (int)julian_start;
    double start_frac = julian_start - start_jd0;
    double end_jd0 = (int)julian_stop;
    double end_frac = julian_stop - end_jd0;

    w_eph = writeph_spk_open("writephspk13.bsp");
    if (!w_eph)
    {  
        printf("Error: opening the spk file failed\n");
        return 1;
    }
    if (!writeph_spk13_seq_write(w_eph, 199, 10, 1, start_jd0, start_frac, end_jd0, end_frac, states2, epochs2, 4056, 9, segid))
    {
        printf("Error: writing the type 13 segment failed\n");
        return 1;
    }
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the spk file failed\n");
        return 1;
    }

    /* add more segments */
    /* open the file */
    w_eph = writeph_spk_open("writephspk13.bsp");
    if (!w_eph)
    {
        printf("Error: opening the spk file failed\n");
        return 1;
    }
    for (j = 0; j < 50; j++)
    {
        if (!writeph_spk13_seq_write(w_eph, 299, 5, 1, 2451545, 0, 2451929, 0, states1, epochs1, 2, 3, segid))
        {
            printf("Error: writing the type 13 segment (index %d) failed\n", j);
            return 1;
        }
    }
    /* close the file */
    if (!writeph_close(w_eph))
    {
        printf("Error: closing the spk file failed\n");
        return 1;
    }

    /* open the file with the calceph module */
    c_eph = calceph_open("writephspk13.bsp");
    if (!c_eph)
    {
        printf("Error: opening the spk file failed\n");
        return 1;
    }

    /* check the coordinates interpolated from the ephemeris file */
    if (!writeph_check_states(c_eph, "writephrefcoordinateslagrange.dat", 199, 10, 1E-6))
    {
        printf("Error: checking the interpolation failed\n");
        return 1;
    }

    calceph_close(c_eph);

    return 0;
}