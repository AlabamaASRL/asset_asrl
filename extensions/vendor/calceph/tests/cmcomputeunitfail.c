/*-----------------------------------------------------------------*/
/*! 
  \file cmcomputeunitfail.c 
  \brief Check the errors of the function calceph_compute_unit.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2011-2021, CNRS
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
 /*printf("msg='%s'\n", msg);*/
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
 int res;
 int target, center;
 double jd0;
 double dt;
 double PV[6];
 t_calcephbin *peph;
 
 calceph_seterrorhandler(3,hidemsg);
 /* open the ephemeris file */
 peph = tests_calceph_open("../examples/example1.dat");
 res = (peph!=NULL);
 if (res)
 {
       res = 0;
       /* read the references coordinates */
       jd0=-9.0721875000000000E+03;
       dt = jd0-(int)jd0;
       jd0 = ((int)jd0)+2.4515450000000000000E+06;
       target=1;
       center=12;
       
       /* try with invalid units */
       if (calceph_compute_unit(peph, jd0, dt, target, center, 0, PV)!=0)
       {
        printf("unit = 0 does not fail\n");
        res = 1;
       }
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_AU+CALCEPH_UNIT_KM+CALCEPH_UNIT_DAY+CALCEPH_UNIT_RAD,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_AU+CALCEPH_UNIT_KM does not fail\n");
        res = 1;
       }
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_KM+CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC+CALCEPH_UNIT_RAD,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_DAY+CALCEPH_UNIT_SEC does not fail\n");
        res = 1;
       }
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_KM+CALCEPH_UNIT_RAD,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_KM+CALCEPH_UNIT_RAD does not fail\n");
        res = 1;
       }
       target=15;
       center=0;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_KM,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_KM and target=15 does not fail\n");
        res = 1;
       }
       target=16;
       center=0;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_KM,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_KM and target=16 does not fail\n");
        res = 1;
       }
       target=17;
       center=0;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_SEC,
                               PV)!=0)
       {
        printf("CALCEPH_UNIT_SEC and target=17 does not fail\n");
        res = 1;
       }
       target=15;
       center=3;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_AU+CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC,
                               PV)!=0)
       {
        printf("target=15 and center!=0 does not fail\n");
        res = 1;
       }
       target=16;
       center=3;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_AU+CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC,
                               PV)!=0)
       {
        printf("target=16 and center!=0 does not fail\n");
        res = 1;
       }
       target=17;
       center=3;
       if (calceph_compute_unit(peph, jd0, dt, target, center, 
                               CALCEPH_UNIT_AU+CALCEPH_UNIT_RAD+CALCEPH_UNIT_SEC,
                               PV)!=0)
       {
        printf("target=17 and center!=0 does not fail\n");
        res = 1;
       }
       
   /* close the ephemeris file */
   calceph_close(peph);
 }
 else res = 1;
 return res;
}
