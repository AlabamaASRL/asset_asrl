/*-----------------------------------------------------------------*/
/*! 
  \file cmparallelcheck.c
  \brief Check the output of the function calceph_compute is thread-safe with an INPOP file.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 

   Copyright, 2019-2026, CNRS
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
#include "openfiles.h"
#if HAVE_CONFIG_H
#include "calcephconfig.h"
#endif
#include "calcephthread.h"

int main(void);

/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
struct threaddata
{
    int *target, *center;

    double *jd0;

    double *dt;

    double *PV;

    t_calcephbin *peph;

    int nlines;

    int threadid;
};

#define NTHREADS 4

/*-----------------------------------------------------------------*/
/*-----------------------------------------------------------------*/
static double myabs(double x);

static int maincheck(int nbfile, const char *const filear[]);

static int countlinesinfiles(const char *filename);

static double myabs(double x)
{
    if (x != x)
        return 1E300;
    return (x > 0 ? x : -x);
}

/*-----------------------------------------------------------------*/
/* count the number of lines  */
/*-----------------------------------------------------------------*/
static int countlinesinfiles(const char *filename)
{
    int j;
    int lines = 0;
    FILE *fcheck;
    double d, jd0;
    int target, center;

    fcheck = tests_fopen(filename, "rt");
    if (fcheck == NULL)
    {
        printf("'%s' is missing\n", filename);
        return 0;
    }
    while (!feof(fcheck))
    {
        int iret = fscanf(fcheck, "%le %d %d ", &jd0, &target, &center);

        if (feof(fcheck))
        {
            break;
        }
        if (iret != 3)
        {
            printf("fscanf failure\n");
            return 0;
        }
        for (j = 0; j < 6; j++)
        {
            if (fscanf(fcheck, "%le", &d) != 1)
            {
                printf("fscanf failure\n");
                return 0;
            }
        }
        lines++;
    }
    fclose(fcheck);
    return lines;
}

/*-----------------------------------------------------------------*/
/* thread computes */
/*-----------------------------------------------------------------*/
define_fn_thrd_start(thread_compute, pthreadata)
{
    int j;
    struct threaddata *pdata = (struct threaddata *) pthreadata;

    for (j = pdata->threadid; j < pdata->nlines; j += NTHREADS)
    {
        if (calceph_compute
            (pdata->peph, pdata->jd0[j], pdata->dt[j], pdata->target[j], pdata->center[j], pdata->PV + 6 * j) == 0)
        {
            printf("calceph_compute fails %d %d\n", j, pdata->threadid);
            return calceph_thrd_error;
        }
    }
    return calceph_thrd_success;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
static int maincheck(int nbfile, const char *const filear[])
{
    const char *filenamecheck = "example1_tests.dat";

    int res = 1, n, j, k;

    int *target, *center;

    double *jd0;

    double *dt;

    double *PV;

    double *PVcheck;

    double seuil;

    FILE *fcheck;

    t_calcephbin *peph;

    int nlines, jline;

    struct threaddata arthreaddata[NTHREADS];
    thrd_start_t fns[NTHREADS];
    void *pargs[NTHREADS];

    union
    {
        unsigned char c[4];
        unsigned int i;
    } endian;

    endian.i = 0x01020304;

    nlines = countlinesinfiles(filenamecheck);

    if (nlines <= 0)
        return 1;

    /* allocate the array */
    target = (int *) malloc(sizeof(int) * nlines);
    center = (int *) malloc(sizeof(int) * nlines);
    jd0 = (double *) malloc(sizeof(double) * nlines);
    dt = (double *) malloc(sizeof(double) * nlines);
    PV = (double *) malloc(sizeof(double) * 6 * nlines);
    PVcheck = (double *) malloc(sizeof(double) * 6 * nlines);

    if (target == NULL || center == NULL || jd0 == NULL || dt == NULL || PV == NULL || PVcheck == NULL)
    {
        printf("can't allocate memory for %d lines\n", nlines);
        return 1;
    }

    fcheck = tests_fopen(filenamecheck, "rt");
    if (fcheck == NULL)
    {
        printf("example1_tests.dat is missing\n");
        return 1;
    }
    /* open the ephemeris file */
    peph = tests_calceph_open_array(nbfile, filear);
    if (peph != NULL)
    {
        if (calceph_isthreadsafe(peph) != 0)
        {
            printf("the function must return 0\n");
            return 1;
        }

        res = calceph_prefetch(peph);
        if (res == 0)
        {
            calceph_close(peph);
            return 1;
        }

        if (calceph_isthreadsafe(peph) == 0)
        {
#if CALCEPH_HAVE_MUTEX
            if (0x04 == endian.c[0])
            {
                printf("the function must return 1 (endian=%d)\n", (int) (endian.c[0]));
                return 1;
            }
            else
#endif
            {                   /* report no error as the endian if the hardware is different of the file */
                return 0;
            }
        }
        /* load the data */
        jline = 0;
        while (!feof(fcheck))
        {
            /* read the references coordinates */
            int iret = fscanf(fcheck, "%le %d %d ", jd0 + jline, target + jline, center + jline);

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
                if (fscanf(fcheck, "%le", PVcheck + 6 * jline + j) != 1)
                {
                    printf("fscanf failure\n");
                    res = 1;
                    break;
                }
            }
            dt[jline] = jd0[jline] - (int) jd0[jline];
            jd0[jline] = ((int) jd0[jline]) + 2.4515450000000000000E+06;
            jline++;
        }

        /*printf("main thread creates and starts other threads\n"); */
        /* perform the computation by multiple threads ( 4 here) */
        for (j = 0; j < NTHREADS; j++)
        {
            arthreaddata[j].target = target;
            arthreaddata[j].center = center;
            arthreaddata[j].jd0 = jd0;
            arthreaddata[j].dt = dt;
            arthreaddata[j].PV = PV;
            arthreaddata[j].peph = peph;
            arthreaddata[j].nlines = nlines;
            arthreaddata[j].threadid = j;
            fns[j] = thread_compute;
            pargs[j] = arthreaddata + j;
        }

        if (calceph_thread_start_and_join(NTHREADS, fns, pargs) == 1)
        {
            printf("calceph_thread_start_and_join fails\n");
            exit(1);
        }

        /*printf("main thread checks the results\n"); */
        /* check the result by a single thread */
        for (jline = 0; jline < nlines; jline++)
        {
            /*printf("%23.16E %23.16E %d %d ", jd0, dt, target, center);
               for (int j=0; j<6; j++) printf("%23.16E ", PVcheck[j]);
               printf("\n"); */
            /* compute  the coordinates */
            seuil = 1E-9;
            /* for libration , return angles between 0 and 2*pi */
            if (target[jline] != 15)
            {
                /* check with references coordinates */
                n = 6;
                for (j = 0; j < n; j++)
                {
                    if (myabs(PVcheck[6 * jline + j] - PV[6 * jline + j]) >= seuil)
                    {
                        printf("failure for %d %d at time %23.16E %23.16E : diff=%e (%23.16E - %23.16E)\n",
                               target[jline], center[jline], jd0[jline], dt[jline],
                               PVcheck[6 * jline + j] - PV[6 * jline + j], PVcheck[6 * jline + j], PV[6 * jline + j]);
                        for (k = 0; k < n; k++)
                            printf("%d : diff=%e (%23.16E - %23.16E)\n", k, PVcheck[6 * jline + k] - PV[6 * jline + k],
                                   PVcheck[6 * jline + k], PV[6 * jline + k]);
                        res = 1;
                        return res;
                    }
                }
            }

        }

        /* close the ephemeris file */
        calceph_close(peph);
    }
    else
        res = 1;
    fclose(fcheck);

    free(target);
    free(center);
    free(jd0);
    free(dt);
    free(PV);
    free(PVcheck);
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

    res = maincheck(1, filearinpop);
    if (res == 0)
        res = maincheck(5, filenamespk);

    return res;
}
