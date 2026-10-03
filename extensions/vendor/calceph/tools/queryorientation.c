/*-----------------------------------------------------------------*/
/*!
  \file queryorientation.c
  \brief It computes for a Julian day TDB the orientation of the target
  \details
   calceph_queryorientation <Julian Day> <Target>  <Files>
   e.g. to compute the orientation of the Moon at JD 2442457 TDB :
     calceph_queryorientation  2442457 301 mykernel.bpc

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2019-2025, CNRS
   email of the author : Mickael.Gastineau@obspm.fr

  History:
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
knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
*/
/*-----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "calceph.h"

static void printcoord(double PV[6]);
int main(int argc, char **argv);

/*-----------------------------------------------------------------*/
/* print coordinates */
/*-----------------------------------------------------------------*/
static void printcoord(double PV[6])
{
    int j;

    /*printf("Units are Radian and Radian/Seconds :\n");
       for (j = 0; j < 6; j++)
       printf("\t%23.16E\n", PV[j]); */
    printf("Units are Radian :\n");
    for (j = 0; j < 3; j++)
        printf("\t%23.16E\n", PV[j]);
    printf("\n");
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(int argc, char **argv)
{
    double PV[6];
    t_calcephbin *peph;
    int target;
    double jd0;

    if (argc < 4)
    {
        printf("calceph_queryorientation - Compute for a Julian day TDB the orientation of the target\n");
        printf("Usage  :  calceph_queryorientation <Julian Day> <Target> <Files...>\n");
        printf("  e.g. :  calceph_queryorientation  2442457 301 example1.bpc\n");
        return 1;
    }
    jd0 = strtod(argv[1], NULL);
    target = atoi(argv[2]);
    /* open the ephemeris file */
    peph = calceph_open_array(argc - 3, (const char *const *) (argv + 3));
    if (peph)
    {
        printf("Time: %23.16f Target: %d \n", jd0, target);

        /* compute and print the coordinates */
        if (calceph_orient_order(peph, jd0, 0., target,
                                 CALCEPH_UNIT_RAD + CALCEPH_UNIT_SEC + CALCEPH_USE_NAIFID, 0, PV) != 0)
        {
            printf("The euler angles: \n");
            printcoord(PV);
        }
        else return 1;
        /* close the ephemeris file */
        calceph_close(peph);
    }
    else
    {
        printf("The ephemeris can't be opened\n");
        return 1;
    }
    return 0;
}
