/*-----------------------------------------------------------------*/
/*! 
  \file cmpolerade.c 
  \brief Check the output of the function calceph_orient_unit if the orientation are provided by a text PCK

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright,  2019, CNRS
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
#define _USE_MATH_DEFINES
#include <math.h>

#include "calceph.h"
#include "openfiles.h"

static double myabs(double x);

static int maincheck(int nbfile, const char *const filear[], const char *filedata);

int main(void);

static double myabs(double x)
{
    if (x != x)
        return 1E300;
    return (x > 0 ? x : -x);
}

/*-----------------------------------------------------------------*/
/* main check function */
/*-----------------------------------------------------------------*/
static int maincheck(int nbfile, const char *const filear[], const char *filedata)
{
    int res, n, j, k;

    int target;

    double jd0;

    double dt;

    double PV[6];

    double PVcheck[6];

    double seuil;

    double myPI = atan2(0., -1.);

    t_calcephbin *peph;

    FILE *fsrc;

    int unit;

    fsrc = tests_fopen(filedata, "rt");
    if (fsrc == NULL)
    {
        printf("example1_tests.dat is missing\n");
        return 1;
    }

    /* open the ephemeris file */
    peph = tests_calceph_open_array(nbfile, filear);
    res = (peph != NULL);

    if (res)
    {
        while (!feof(fsrc))
        {
            int iret = fscanf(fsrc, "%le %d", &jd0, &target);

            if (feof(fsrc))
            {
                res = 0;
                break;
            }
            if (iret != 2)
            {
                printf("fscanf failure (iret=%d)\n", iret);
                res = 1;
                break;
            }
            for (j = 0; j < 3; j++)
            {
                if (fscanf(fsrc, "%le", PV + j) != 1)
                {
                    printf("fscanf failure - line %f %d\n", jd0, target);
                    res = 1;
                    break;
                }
            }
            jd0 /= 86400.;
            dt = jd0 - (int) jd0;
            jd0 = ((int) jd0) + 2.4515450000000000000E+06;

            /*printf("%23.16E %23.16E %d %d ", jd0, dt, target, center);
               for (int j=0; j<6; j++) printf("%23.16E ", PVcheck[j]);
               printf("\n"); */
            /* compute  the coordinates */
            seuil = 1E-9;

            unit = CALCEPH_UNIT_RAD + CALCEPH_UNIT_DAY + CALCEPH_USE_NAIFID;
            n = 6;
            for (j = 0; j < n; j++)
                PVcheck[j] = 0.E0;
            if (calceph_orient_order/*unit*/(peph, jd0, dt, target, unit, 0, PVcheck) == 0)
            {
                res = 1;
                break;
            }
            PV[0] = myPI / 2 + PV[0];
            PV[1] = myPI / 2. - PV[1];

            n = 3;
            for (j = 0; j < n; j++)
            {
                double x = fmod(fmod(PVcheck[j], 2. * myPI) + 2. * myPI, 2. * myPI);
                double y = fmod(fmod(PV[j], 2. * myPI) + 2. * myPI, 2. * myPI);

                if (myabs(x - y) >= seuil)
                {
                    printf("failure  for %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\nPV:\n",
                           target, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                    for (k = 0; k < 3; k++)
                    {
                        printf("diff=%e (%23.16E - %23.16E)\n", PVcheck[k] - PV[k], PVcheck[k], PV[k]);
                    }
                    printf("mod 2pi=%e (%23.16E - %23.16E)\n", x - y, x, y);
                    res = 1;
                    return res;
                }
            }
        }

        /* close the ephemeris file */
        calceph_close(peph);
    }
    else
        res = 1;
    fclose(fsrc);
    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    int res = 0;
    const char *filenamespk[] = { "../tests/checkpolerade.tpc" };
    res = maincheck(1, filenamespk, "../tests/checkpolerade_499.dat");
    if (res == 0)
        res = maincheck(1, filenamespk, "../tests/checkpolerade_599.dat");
    if (res == 0)
        res = maincheck(1, filenamespk, "../tests/checkpolerade_501.dat");
    return res;
}
