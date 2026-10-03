/*-----------------------------------------------------------------*/
/*! 
  \file crotangmomorder0.c 
  \brief Check the output of the function calceph_rotangmom_order with order = 0.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2017, 2018, 2019,  CNRS
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
#include "calceph.h"
#include "openfiles.h"

static double myabsrel(double x, double y);
static double myabs(double x);

static int maincheck(int nbfile, const char *const filear[], int prefetch);

int main(void);

static double myabs(double x)
{
    if (x != x)
        return 1E300;
    return (x > 0 ? x : -x);
}

static double myabsrel(double x, double y)
{
    if (x != x)
        return 1E300;
    if (y != y)
        return 1E300;
    if (myabs(x) > 1E-50)
    {
        if (myabs(y / x) - 1E0 > 1E-2)
            return 1E200;
    }
    return (myabs(x) < 1E-10 ? myabs(x - y) : myabs((x - y) / x));
}

/*-----------------------------------------------------------------*/
/* main check function */
/*-----------------------------------------------------------------*/
static int maincheck(int nbfile, const char *const filear[], int prefetch)
{
    int res, n, j;

    int target, center;

    double jd0;

    double dt;

    double PV[3];

    double PVcheck[3];

    double seuil;

    FILE *fcheck;

    t_calcephbin *peph;

    int unit;

    fcheck = tests_fopen("example2_rotangmom_tests.dat", "rt");
    if (fcheck == NULL)
    {
        printf("example2_rotangmom_tests.dat is missing\n");
        return 1;
    }
    /* open the ephemeris file */
    peph = tests_calceph_open_array(nbfile, filear);
    res = (peph != NULL);

    if (prefetch == 1 && res)
    {
        res = calceph_prefetch(peph);
        if (res == 0)
            calceph_close(peph);
    }

    if (res)
    {
        while (!feof(fcheck))
        {
            /* read the references coordinates */
            int iret = fscanf(fcheck, "%le %d %d ", &jd0, &target, &center);

            if (feof(fcheck))
            {
                res = 0;
                break;
            }
            if (iret != 3)
            {
                printf("fscanf failure\n");
                res = 1;
            }
            for (j = 0; j < 3; j++)
            {
                if (fscanf(fcheck, "%le", PVcheck + j) != 1)
                {
                    printf("fscanf failure\n");
                    res = 1;
                    break;
                }
            }
            dt = jd0 - (int) jd0;
            jd0 = ((int) jd0) + 2.4515450000000000000E+06;

            /*printf("%23.16E %23.16E %d %d ", jd0, dt, target, center);
               for (j=0; j<6; j++) printf("%23.16E ", PVcheck[j]);
               printf("\n"); */

            /* check with references coordinates */
            n = 3;
            seuil = 3E-10;

            /* check DAY */
            unit = CALCEPH_UNIT_DAY;
            if (target > 100)
                unit += CALCEPH_USE_NAIFID;
            for (j = 0; j < n; j++)
                PV[j] = 0.E0;
            if (calceph_rotangmom_order(peph, jd0, dt, target, unit, 0, PV) == 0)
            {
                res = 1;
                break;
            }

            for (j = 0; j < n; j++)
            {
                if (myabsrel(PVcheck[j], PV[j]) >= seuil)
                {
                    printf("failure DAY for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                           target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                    printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                    res = 1;
                    return res;
                }
            }

            /* check SEC */
            unit = CALCEPH_UNIT_SEC;
            if (target > 100)
                unit += CALCEPH_USE_NAIFID;
            for (j = 0; j < n; j++)
                PV[j] = 0.E0;

            if (calceph_rotangmom_order(peph, jd0, dt, target, unit, 0, PV) == 0)
            {
                res = 1;
                break;
            }

            for (j = 0; j < n; j++)
            {
                if (myabsrel(PVcheck[j] / 86400.E0, PV[j]) >= seuil)
                {
                    printf("failure SEC for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                           target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                    printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
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
    fclose(fcheck);
    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    int res = 0;
    const char *filearinpop[] = { "../examples/example2_rotangmom.dat" };

    res = maincheck(1, filearinpop, 0);
    if (res == 0)
        res = maincheck(1, filearinpop, 1);
    return res;
}
