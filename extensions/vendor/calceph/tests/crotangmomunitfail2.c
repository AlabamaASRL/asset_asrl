/*-----------------------------------------------------------------*/
/*! 
  \file crotangmomunitfail2.c 
  \brief Check the errors of the function calceph_rotangmom_unit 
   with a file that contains G(mR^2) of the Earth.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2017, CNRS
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
    /* printf("msg='%s'\n", msg); */
}

/*-----------------------------------------------------------------*/
/* check the errors with file */
/*-----------------------------------------------------------------*/
static int checkfile(const char *filename)
{
    int res;

    double jd0;

    double dt;

    double PV[6];

    int k;

    t_calcephbin *peph;

    calceph_seterrorhandler(3, hidemsg);
    /* open the ephemeris file */
    peph = tests_calceph_open(filename);
    res = (peph != NULL);
    if (res)
    {
        res = 0;
        dt = 0.E0;
        jd0 = 2451545.00000000-100;

        /* try with invalid targets */
        for (k = 0; k < 17; k++)
        {
            if (k==3) continue;
            if (calceph_rotangmom_unit(peph, jd0, dt, k, CALCEPH_UNIT_DAY, PV) != 0)
            {
                printf("target=%d does not fail for '%s'\n", k, filename);
                res = 1;
            }
            if (calceph_rotangmom_unit(peph, jd0, dt, k*100+99, CALCEPH_UNIT_DAY+CALCEPH_USE_NAIFID, PV) != 0)
            {
                printf("target=%d does not fail for '%s'\n", k*100+99, filename);
                res = 1;
            }
        }
        
        /* invalid unit */
        if (calceph_rotangmom_unit(peph, jd0, dt, 3, 0, PV) != 0)
        {
            printf("unit=0 does not fail for '%s'\n", filename);
            res = 1;
        }
        if (calceph_rotangmom_unit(peph, jd0, dt, 399, CALCEPH_USE_NAIFID, PV) != 0)
        {
            printf("unit=CALCEPH_USE_NAIFID does not fail for '%s'\n", filename);
            res = 1;
        }
        if (calceph_rotangmom_unit(peph, jd0, dt, 3, CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC, PV) != 0)
        {
            printf("unit=0 does not fail for '%s'\n", filename);
            res = 1;
        }
        if (calceph_rotangmom_unit(peph, jd0, dt, NAIFID_EARTH, CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC+CALCEPH_USE_NAIFID, PV) != 0)
        {
            printf("unit=CALCEPH_USE_NAIFID does not fail for '%s'\n", filename);
            res = 1;
        }
        
        /*invalid time */
        if (calceph_rotangmom_unit(peph, 0.E0, 0.E0, 3, CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC, PV) != 0)
        {
            printf("time=0 does not fail for '%s'\n", filename);
            res = 1;
        }
        if (calceph_rotangmom_unit(peph, 0.E0, 0.E0, NAIFID_EARTH, CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC+CALCEPH_USE_NAIFID, PV) != 0)
        {
            printf("time=0  and CALCEPH_USE_NAIFID does not fail for '%s'\n", filename);
            res = 1;
        }
       
        /* close the ephemeris file */
        calceph_close(peph);
    }
    else
        res = 1;
    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    return checkfile("../examples/example2_rotangmom.dat");
}
