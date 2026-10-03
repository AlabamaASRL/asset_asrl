/*-----------------------------------------------------------------*/
/*! 
  \file cmbecnh.c 
  \brief Benchmark the function SPICE NAIF vs calceph_compute_order.
  calceph_compute_order is the same as calceph_compute_unit if order=1

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2014, 2015, 2016, 2017, CNRS
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
#include <sys/resource.h>
#include "calceph.h"
/*#define HAVE_SPICE 1*/
#if HAVE_SPICE
#include "SpiceUsr.h"
#endif

static void maincheck(int nbfile, const char *const filear[], const char *format, int prefetch, int nday, double offset, int order);
static const char *order_to_str(int order);

int main(int argc, char **argv);

#define HAVE_GETRUSAGE 1

static unsigned long calceph_cputime(void)
{
#ifdef HAVE_GETRUSAGE
    struct rusage ru;

    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec * 1000000 + ru.ru_utime.tv_usec + ru.ru_stime.tv_sec * 1000000 + ru.ru_stime.tv_usec;
#else
    printf("\ngetrusage not available\n");
    exit(1);
    return 0;
#endif
}

/*-----------------------------------------------------------------*/
/* return the string associated to the order */
/*-----------------------------------------------------------------*/
static const char *order_to_str(int order)
{
    static const char* szorder[4] = { "P___", "PV__", "PVA_", "PVAJ" };
    return szorder[order];
}

/*-----------------------------------------------------------------*/
/* main check function */
/*-----------------------------------------------------------------*/
static void maincheck(int nbfile, const char *const filear[], const char *format, int prefetch, int nday, double offset, int order)
{
    int j, k;

    int target, center;

    double jd0;

    double dt;

    unsigned long tstart, tend;

    t_calcephbin *peph;

    int unit;

#define N 100000
    double ardt[N];

    double arjd0[N];

    double *arPVcheck;

    
    arPVcheck = (double*)malloc(sizeof(double)*(12 * N));
    if (arPVcheck==NULL)
    {
        printf("can't allocate memory for benchhmark\n");
        exit(1);
    }
    
    /* open the ephemeris file */
    peph = calceph_open_array(nbfile, filear);
    if (peph)
    {
        if (prefetch == 1)
            calceph_prefetch(peph);
        dt = 0.E0;
        jd0 = 2.4515450000000000000E+06 + offset;
        for (j = 0; j < N; j++)
        {
            arjd0[j] = jd0 + (j % nday) - nday / 2;
            ardt[j] = dt + (j * 24.E0) / N;
            while (ardt[j] > 1.E0)
                ardt[j]--;
            while (ardt[j] < -1.E0)
                ardt[j]--;
        }
        unit = CALCEPH_UNIT_KM + CALCEPH_UNIT_SEC + CALCEPH_USE_NAIFID;
        target = NAIFID_JUPITER_BARYCENTER;
        center = NAIFID_SOLAR_SYSTEM_BARYCENTER;
        tstart = calceph_cputime();
#define NITER 10
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf
            ("calceph_compute_order %s  on %s file : %8.0f evaluations per second : %s of SOLAR_SYSTEM_BARYCENTER-->JUPITER_BARYCENTER and linear access\n",
             prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6, order_to_str(order));
        target = NAIFID_JUPITER_BARYCENTER;
        center = NAIFID_EARTH_MOON_BARYCENTER;
        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf
            ("calceph_compute_order %s on %s file : %8.0f evaluations per second :%s of  EARTH MOON BARYCENTER -->JUPITER_BARYCENTER and linear access\n",
             prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6, order_to_str(order));

        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                target = random() % 10;
                center = random() % 10;

                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf("calceph_compute_order %s on %s file : %8.0f evaluations per second : %s of random center-->random target and linear access\n",
               prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6,  order_to_str(order));

        /* randomize the time */
        for (j = 0; j < N; j++)
        {
            arjd0[j] = jd0 + (random() % nday) - nday / 2;
        }
 
        target = NAIFID_JUPITER_BARYCENTER;
        center = NAIFID_SOLAR_SYSTEM_BARYCENTER;
        tstart = calceph_cputime();
#define NITER 10
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf
            ("calceph_compute_order %s on %s file : %8.0f evaluations per second : %s of SOLAR_SYSTEM_BARYCENTER-->JUPITER_BARYCENTER and random time\n",
             prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6, order_to_str(order));
        target = NAIFID_JUPITER_BARYCENTER;
        center = NAIFID_EARTH_MOON_BARYCENTER;
        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf
            ("calceph_compute_order %s on %s file : %8.0f evaluations per second : %s of EARTH MOON BARYCENTER -->JUPITER_BARYCENTER and random time\n",
             prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6, order_to_str(order));

 
        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                target = random() % 10;
                center = random() % 10;

                if (calceph_compute_order(peph, arjd0[k], ardt[k], target, center, unit, order, arPVcheck + 12 * k) == 0)
                {
                    return;
                }
            }
        }
        tend = calceph_cputime();
        printf
            ("calceph_compute_order %s on %s file : %8.0f evaluations per second : %s ofrandom center-->random target and random time\n",
             prefetch == 0 ? "" : "with prefetch", format, ((double) (N * NITER)) / (tend - tstart) * 1E6, order_to_str(order));

        /* close the ephemeris file */
        calceph_close(peph);
    }
    free(arPVcheck);
}

#if HAVE_SPICE
/*-----------------------------------------------------------------*/
/* main check function with spice software */
/*-----------------------------------------------------------------*/
static void maincheckspice(int nbfile, const char *const filear[], const char *format, int nday, double offset)
{
    int j, k, l;

    double jd0;

    double dt;

    unsigned long tstart, tend;

    int target, center;

    double ardt[N];

    double arjd0[N];

    double arPVcheck[6 * N];

    const char *objectname[] = { "SSB",
        "MERCURY BARYCENTER",
        "VENUS BARYCENTER",
        "EMB",
        "MARS BARYCENTER",
        "JUPITER BARYCENTER",
        "SATURN BARYCENTER",
        "URANUS BARYCENTER",
        "NEPTUNE BARYCENTER",
        "PLUTO BARYCENTER",
        "SUN"
    };

    for (k = 0; k < nbfile; k++)
        furnsh_c(filear[k]);

    dt = 0.E0;
    jd0 = 2.4515450000000000000E+06 + offset;
    for (l = 0; l < 1; l++)
    {
        for (j = 0; j < N; j++)
        {
            arjd0[j] = jd0 + (j % nday) - nday / 2;
            ardt[j] = dt + (j * 24.E0) / N;
            while (ardt[j] > 1.E0)
                ardt[j]--;
            while (ardt[j] < -1.E0)
                ardt[j]--;
        }
        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                spkezr_c("JUPITER BARYCENTER", (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000",
                         "NONE", "SOLAR SYSTEM BARYCENTER", arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : SOLAR_SYSTEM_BARYCENTER-->JUPITER_BARYCENTER and linear access\n",
               format, ((double) (N * NITER)) / (tend - tstart) * 1E6);

        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                spkezr_c("JUPITER BARYCENTER", (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000",
                         "NONE", "EARTH MOON BARYCENTER", arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : EARTH MOON BARYCENTER-->JUPITER_BARYCENTER and linear access\n",
               format, ((double) (N * NITER)) / (tend - tstart) * 1E6);

        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                target = random() % 10;
                center = random() % 10;

                spkezr_c(objectname[target], (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000", "NONE",
                         objectname[center], arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : random center-->random target and linear access\n", format,
               ((double) (N * NITER)) / (tend - tstart) * 1E6);

        /* randomize time */
        for (j = 0; j < N; j++)
        {
            arjd0[j] = jd0 + (random() % nday) - nday / 2;
        }
        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                spkezr_c("JUPITER BARYCENTER", (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000",
                         "NONE", "SOLAR SYSTEM BARYCENTER", arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : SOLAR_SYSTEM_BARYCENTER-->JUPITER_BARYCENTER and random time\n",
               format, ((double) (N * NITER)) / (tend - tstart) * 1E6);

        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                spkezr_c("JUPITER BARYCENTER", (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000",
                         "NONE", "EARTH MOON BARYCENTER", arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : EARTH MOON BARYCENTER-->JUPITER_BARYCENTER and random time\n",
               format, ((double) (N * NITER)) / (tend - tstart) * 1E6);

        tstart = calceph_cputime();
        for (j = 0; j < NITER; j++)
        {
            for (k = 0; k < N; k++)
            {
                double LT;

                target = random() % 10;
                center = random() % 10;

                spkezr_c(objectname[target], (arjd0[k] + ardt[k] - 2.4515450000000000000E+06) * 86400, "J2000", "NONE",
                         objectname[center], arPVcheck + 6 * k, &LT);
            }
        }
        tend = calceph_cputime();
        printf("spkezr_c on %s file : %8.0f evaluations per second : random center-->random target and random time\n",
               format, ((double) (N * NITER)) / (tend - tstart) * 1E6);
    }
}
#endif

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(int argc, char **argv)
{
    int res = 0;

    int day;
    
    double  offset;
    
    int order;

    if (argc == 3)
    {
        const char *filear[1];

        int day = atoi(argv[2]);
        
        offset = 0.E0;

        printf("benchmark done with the file '%s' on %d days around J2000 \n", argv[1], day);
        filear[0] = argv[1];

#if HAVE_SPICE
        maincheckspice(1, filear, "SPICE", day, offset);
#endif
        for (order=0; order<=3; order++)
        {
            maincheck(1, filear, "SPICE", 0, day, offset, order);
            maincheck(1, filear, "SPICE", 1, day, offset, order);
        }    
    }
    else
    {
        const char *filearinpop[] = { "../examples/example1.dat" };
        const char *filenamespk[] = { "../examples/example1.bsp", "../examples/example1.tpc",
            "../examples/example1.bpc", "../examples/example1spk_time.bsp"
        };

        day = 1000;
        offset  = -4.8756250000000000E+03 ;
#if HAVE_SPICE
        maincheckspice(4, filenamespk, "SPICE", day, offset);
#endif
        for (order=0; order<=3; order++)
        {
            maincheck(1, filearinpop, "INPOP/old JPL", 0, day, offset, order);
            maincheck(1, filearinpop, "INPOP/old JPL", 1, day, offset, order);
            maincheck(4, filenamespk, "SPICE", 0, day, offset, order);
            maincheck(4, filenamespk, "SPICE", 1, day, offset, order);
        }    
    }
    return res;
}
