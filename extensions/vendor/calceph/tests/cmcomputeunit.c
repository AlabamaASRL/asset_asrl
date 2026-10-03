/*-----------------------------------------------------------------*/
/*! 
  \file cmcomputeunit.c 
  \brief Check the output of the function calceph_compute_unit.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,  CNRS
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

static double myabs(double x);

static int maincheck(int nbfile, const char *const filear[], int prefetch);

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
static int maincheck(int nbfile, const char *const filear[], int prefetch)
{
    int res, n, j, k;

    int target, center;

    double jd0;

    double dt;

    double PV[6];

    double PV2[6];

    double PVcheck[6];

    double seuil;

    double AU;

    FILE *fcheck;

    t_calcephbin *peph;

    int unit;

    fcheck = tests_fopen("example1_tests.dat", "rt");
    if (fcheck == NULL)
    {
        printf("example1_tests.dat is missing\n");
        return 1;
    }
    /* open the ephemeris file */
    peph = tests_calceph_open_array(nbfile, filear);
    res = (peph != NULL);

    if (res)
    {
        res = calceph_getconstant(peph, "AU", &AU);
        if (prefetch == 1)
        {
            res = calceph_prefetch(peph);
            if (res == 0)
                calceph_close(peph);
        }
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
                break;
            }
            for (j = 0; j < 6; j++)
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
               for (int j=0; j<6; j++) printf("%23.16E ", PVcheck[j]);
               printf("\n"); */
            /* compute  the coordinates */

            if (calceph_compute(peph, jd0, dt, target, center, PV) == 0)
            {
                printf("prefetch=%d\n", prefetch);
                res = 1;
                break;
            }
            for (j = 0; j < 6; j++)
                PV2[j] = PV[j];

            seuil = 1E-8;
            /* for libration , return angles between 0 and 2*pi */
            if (target == 15)
            {
                seuil = 1E-7;
                for (j = 2; j <= 2; j++)
                {
                    double pi2 = 6.283185307179586E0;

                    while (PV2[j] >= pi2)
                        PV2[j] = PV2[j] - pi2;
                    while (PV2[j] <= 0)
                        PV2[j] = PV2[j] + pi2;
                }
            }

            /* check with references coordinates */
            n = 6;
            for (j = 0; j < n; j++)
            {
                if (myabs(PVcheck[j] - PV2[j]) >= seuil)
                {
                    printf("failure for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                           target, center, jd0, dt, PVcheck[j] - PV2[j], PVcheck[j], PV2[j]);
                    for (k = 0; k < n; k++)
                        printf("%d : diff=%e (%23.16E - %23.16E)\n", k, PVcheck[k] - PV2[k], PVcheck[k], PV2[k]);
                    printf("prefetch=%d\n", prefetch);
                    res = 1;
                    return res;
                }
            }

            seuil = 3E-15;
            if (target != 15 && target != 16)
            {
                /* check UA/DAY */
                unit = CALCEPH_UNIT_AU + CALCEPH_UNIT_DAY;
                n = 6;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure AU/DAY for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
                /* check UA/SEC */
                unit = CALCEPH_UNIT_AU + CALCEPH_UNIT_SEC;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                for (j = 3; j < n; j++)
                    PVcheck[j] *= 86400E0;
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure AU/SEC for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
                seuil = 3E-14;
                /* check KM/DAY */
                unit = CALCEPH_UNIT_KM + CALCEPH_UNIT_DAY;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                for (j = 0; j < n; j++)
                    PVcheck[j] /= AU;
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure KM/DAY for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
                /* check KM/SEC */
                unit = CALCEPH_UNIT_KM + CALCEPH_UNIT_SEC;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                for (j = 0; j < n; j++)
                    PVcheck[j] /= AU;
                for (j = 3; j < n; j++)
                    PVcheck[j] *= 86400E0;
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure KM/SEC for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
            }

            if (target == 15)
            {
                /* check RAD/DAY */
                unit = CALCEPH_UNIT_RAD + CALCEPH_UNIT_DAY;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure RAD/DAY for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
                /* check AU/SEC */
                unit = CALCEPH_UNIT_RAD + CALCEPH_UNIT_SEC;
                for (j = 0; j < n; j++)
                    PVcheck[j] = 0.E0;
                if (calceph_compute_unit(peph, jd0, dt, target, center, unit, PVcheck) == 0)
                {
                    res = 1;
                    break;
                }
                for (j = 3; j < n; j++)
                    PVcheck[j] *= 86400E0;
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[j] - PV[j]) >= seuil)
                    {
                        printf("failure RAD/SEC for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target, center, jd0, dt, PVcheck[j] - PV[j], PVcheck[j], PV[j]);
                        printf("prefetch=%d file='%s'\n", prefetch, filear[0]);
                        res = 1;
                        return res;
                    }
                }
            }
        };

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
    const char *filearinpop[] = { "../examples/example1.dat" };
    const char *filenamespk[] = { "../examples/example1.bsp", "../examples/example1.tpc",
        "../examples/example1.tf", "../examples/example1.bpc", "../examples/example1spk_time.bsp"
    };

    res = maincheck(1, filearinpop, 0);
    if (res == 0)
        res = maincheck(5, filenamespk, 0);
    if (res == 0)
        res = maincheck(1, filearinpop, 1);
    if (res == 0)
        res = maincheck(5, filenamespk, 1);
    return res;
}
