/*-----------------------------------------------------------------*/
/*! 
  \file cmorientunitfail_11612.c
  \brief Check if the function calceph_compute_unit fails 
  if the unit RADIAN isnot specified :
  Check the bug [#11612] calceph_orient_unit and calceph_orient_order return always 1 even if there are errors

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2018, 2019,  CNRS
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
#include "calcephconfig.h"
#include "openfiles.h"

static int maincheck(int nbfile, const char *const filear[]);

static void hidemsg(const char *msg);

int main(void);

/*-----------------------------------------------------------------*/
/* function to hide the error message */
/*-----------------------------------------------------------------*/
static void hidemsg(const char *PARAMETER_UNUSED(msg))
{
#if HAVE_PRAGMA_UNUSED
#pragma unused(msg)
#endif
    /*   printf("msg='%s'\n", msg); */
}

/*-----------------------------------------------------------------*/
/* main check function */
/*-----------------------------------------------------------------*/
static int maincheck(int nbfile, const char *const filear[])
{
    int res;

    double jd0;

    double dt;

    double PVcheck[6];

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
    if (peph != NULL)
    {
        int iret = fscanf(fcheck, "%le", &jd0);

        if (feof(fcheck))
        {
            res = 0;
        }
        else if (iret != 1)
        {
            printf("fscanf failure\n");
            res = 1;
        }
        dt = jd0 - (int) jd0;
        jd0 = ((int) jd0) + 2.4515450000000000000E+06;
        unit = CALCEPH_UNIT_DAY;
        res = calceph_orient_unit(peph, jd0, dt, NAIFID_MOON, unit + CALCEPH_USE_NAIFID, PVcheck);
        if (res == 0)
        {
            res = calceph_orient_order(peph, jd0, dt, NAIFID_MOON, unit + CALCEPH_USE_NAIFID, 0, PVcheck);
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
    const char *filenamespk1[] = { "../examples/example1.bsp", "../examples/example1.tpc",
        "../examples/example1.tf", "../examples/example1.bpc", "../examples/example1spk_time.bsp"
    };

    const char *filenamespk2[] = { "../examples/example1.dat" };

    calceph_seterrorhandler(3, hidemsg);
    if (maincheck(4, filenamespk1) == 1)
        return 1;
    if (maincheck(1, filenamespk2) == 1)
        return 1;
    return 0;
}
