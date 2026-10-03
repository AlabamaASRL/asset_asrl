/*-----------------------------------------------------------------*/
/*! 
  \file cmaxsupportedderivative.c 
  \brief Check the output of the function calceph_getmaxsupportedorder.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2021-2026,  CNRS
   email of the author : Mickael.Gastineau@obspm.fr

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

static int maincheck(const char *filename, int idseg, int expected, double jd, int target, int center);
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
}

/*-----------------------------------------------------------------*/
/* main check function */
/*-----------------------------------------------------------------*/
static int maincheck(const char *filename, int idseg, int expected, double jd, int target, int center)
{
    int res = 0;
    int order;
    int unit = CALCEPH_UNIT_KM + CALCEPH_UNIT_SEC + CALCEPH_USE_NAIFID;
    double PV[12];

    t_calcephbin *peph;

    order = calceph_getmaxsupportedorder(idseg);
    if (order != expected)
    {
        printf("checking %s %d ...\n", filename, idseg);
        printf("calceph_getmaxsupportedorder(%d) returns %d, but expected %d\n", idseg, order, expected);
        return 1;
    }

    /* open the ephemeris file */
    peph = tests_calceph_open(filename);

    if (peph != NULL)
    {
        if (calceph_compute_order(peph, jd, 0., target, center, unit, expected, PV) == 0)
        {
            printf("calceph_compute_order(%d, order) returns  0\n", idseg);
            res = 1;
        }
        if (expected < 3)
        {
            if (calceph_compute_order(peph, jd, 0., target, center, unit, expected + 1, PV) != 0)
            {
                printf("calceph_compute_order(%d, order+1) returns != 0\n", idseg);
                res = 1;
            }
        }
        calceph_close(peph);
    }

    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    int res = 0;

    calceph_seterrorhandler(3, hidemsg);

    res = maincheck("../examples/example1.dat", CALCEPH_SEGTYPE_ORIG_0, 3, 2442460., 1, 0);
    if (res == 0)
        res = maincheck("../examples/example1.bsp", CALCEPH_SEGTYPE_SPK_3, 3, 2442460., 1, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg1.bsp", CALCEPH_SEGTYPE_SPK_1, 3, 2451545.50, 2000001, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg8.bsp", CALCEPH_SEGTYPE_SPK_8, 3, 2450586.00, 4, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg9.bsp", CALCEPH_SEGTYPE_SPK_9, 3, 2450106.00, 1, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg12.bsp", CALCEPH_SEGTYPE_SPK_12, 3, 2450586.00, 4, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg13.bsp", CALCEPH_SEGTYPE_SPK_13, 3, 2450106.00, 1, 0);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg14.bsp", CALCEPH_SEGTYPE_SPK_14, 3, 2451545.50, 2000, 10);
    if (res == 0)
        res = maincheck("../tests/example1spk_seg21.bsp", CALCEPH_SEGTYPE_SPK_21, 3, 2458601.00, 2065803, 0);
    return res;
}
