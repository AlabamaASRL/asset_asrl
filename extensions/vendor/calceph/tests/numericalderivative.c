/*-----------------------------------------------------------------*/
/*! 
  \file numericalderivative.c 
  \brief Compute an approximation of the derivative function for the tests.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2016-2021, CNRS
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

#include "calcephconfig.h"
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#include "calceph.h"
#include "numericalderivative.h"

/* static values for myfunc_dfridr */
static t_calcephbin *s_eph = NULL;

static double s_JD0;

static int s_target = -1;

static int s_center = -1;

static int s_unit = -1;

static int s_indcomp = -1;

/*-----------------------------------------------------------------*/
/* return abs(x)
 */
/*-----------------------------------------------------------------*/
static double myabs(double x)
{
    if (x != x)
        return 1E300;
    return (x > 0 ? x : -x);
}

/*-----------------------------------------------------------------*/
/* return max(x,y)
 */
/*-----------------------------------------------------------------*/
static double mymax(double x, double y)
{
    return (x > y ? x : y);
}

/*-----------------------------------------------------------------*/
/* based on dfridr : Numercial Recipes p 188/189 (Chapter 5. Evaluation of Functions)
 */
/*-----------------------------------------------------------------*/
static double dfridr(double (*func) (double), double x, double h, double *err)
{
#define CON 1.4E0
#define CON2 (CON*CON)
#define BIG 1.0e30
#define NTAB 10
#define SAFE 2.0E0
    int i, j;

    double errt, fac,  hh, ans;

    double a[NTAB + 1][NTAB + 1];

    ans = -1E300;
    if (h == 0.E0)
        return -1E300;
    hh = h;
    a[1][1] = ((*func) (x + hh) - (*func) (x - hh)) / (2.0E0 * hh);
    *err = BIG;
    for (i = 2; i <= NTAB; i++)
    {
        hh /= CON;
        a[1][i] = ((*func) (x + hh) - (*func) (x - hh)) / (2.0E0 * hh);
        fac = CON2;
        for (j = 2; j <= i; j++)
        {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - 1.0E0);
            fac = CON2 * fac;
            errt = mymax(myabs(a[j][i] - a[j - 1][i]), myabs(a[j][i] - a[j - 1][i - 1]));
            if (errt <= *err)
            {
                *err = errt;
                ans = a[j][i];
            }
        }
        if (myabs(a[i][i] - a[i - 1][i - 1]) >= SAFE * (*err))
            break;
    }
    return ans;
}

/*-----------------------------------------------------------------*/
/* evaluate the first derivative function using calceph_compute_unit at time (since JD0)
*/
/*-----------------------------------------------------------------*/
static double myfunc_first_derivative_compute(double time)
{
    double PV[6];
    double factunit = 1.E0;
    if ((s_unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;
    calceph_compute_unit(s_eph, s_JD0, time/factunit, s_target, s_center, s_unit, PV);
    return PV[3 + s_indcomp];
}

/*-----------------------------------------------------------------*/
/* evaluate the second derivative function 
 using myfunc_first_derivative_compute at time (since JD0)
*/
/*-----------------------------------------------------------------*/
static double myfunc_second_derivative_compute(double time)
{
    double err;
    double factunit = 1.E0;
    if ((s_unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;

    return dfridr(myfunc_first_derivative_compute, time, 1E-1*factunit, &err);
}

/*-----------------------------------------------------------------*/
/* evaluate the first and second derivative unsing calceph_compute_unit and dfridr 
*/
/*-----------------------------------------------------------------*/
int tests_calceph_evaluate_derivate_compute(t_calcephbin * eph, double JD0, double time, int target,
                                    int center, int unit, double PV[6])
{
    double err;
    double factunit = 1.E0;

    int j;

    s_eph = eph;
    s_JD0 = JD0;
    s_target = target;
    s_center = center;
    s_unit = unit;
    if ((unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;
    for (j = 0; j < 3; j++)
    {
        s_indcomp = j;
        PV[j] = dfridr(myfunc_first_derivative_compute, time*factunit, 1E-1*factunit, &err);
        PV[j + 3] = dfridr(myfunc_second_derivative_compute, time*factunit, 1E-1*factunit, &err);
    }
    return (err<=1);
}


/*-----------------------------------------------------------------*/
/* evaluate the  function 
using calceph_rotangmom_unit at time (since JD0)
*/
/*-----------------------------------------------------------------*/
static double myfunc_rotangmom(double time)
{
    double PV[6];
    double factunit = 1.E0;
    if ((s_unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;
    calceph_rotangmom_unit(s_eph, s_JD0, time/factunit, s_target, s_unit, PV);
    return PV[s_indcomp];
}


/*-----------------------------------------------------------------*/
/* evaluate the first derivative function 
using calceph_rotangmom_unit at time (since JD0)
*/
/*-----------------------------------------------------------------*/
static double myfunc_first_derivative_rotangmom(double time)
{
    double err;
    double factunit = 1.E0;
    if ((s_unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;

    return dfridr(myfunc_rotangmom, time, 1E-1*factunit, &err);
}

/*-----------------------------------------------------------------*/
/* evaluate the second derivative function 
using myfunc_first_derivative_rotangmom at time (since JD0)
*/
/*-----------------------------------------------------------------*/
static double myfunc_second_derivative_rotangmom(double time)
{
    double err;
    double factunit = 1.E0;
    if ((s_unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;

    return dfridr(myfunc_first_derivative_rotangmom, time, 1E-1*factunit, &err);
}

/*-----------------------------------------------------------------*/
/* evaluate the first and second and third derivative 
unsing calceph_rotangmom_unit and dfridr 
*/
/*-----------------------------------------------------------------*/
int tests_calceph_evaluate_derivate_rotangmom(t_calcephbin * eph, double JD0, double time, int target,
                                     int unit, double PV[9])
{
    double err;
    double factunit = 1E0;

    int j;

    s_eph = eph;
    s_JD0 = JD0;
    s_target = target;
    s_unit = unit;
    if ((unit&CALCEPH_UNIT_SEC)!=0) factunit=86400E0;
    for (j = 0; j < 3; j++)
    {
        s_indcomp = j;
        PV[j] = dfridr(myfunc_rotangmom, time*factunit, 1E-1*factunit, &err);
        PV[j + 3] = dfridr(myfunc_first_derivative_rotangmom, time*factunit, 1E-1*factunit, &err);
        PV[j + 6] = dfridr(myfunc_second_derivative_rotangmom, time*factunit, 1E-1*factunit, &err);
    }
    return (err<=1);
}
