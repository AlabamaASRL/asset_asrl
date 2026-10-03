/*-----------------------------------------------------------------*/
/*! 
  \file cmgetconstant_11627.c 
  \brief Check that calceph_getconstant returns the number of values 
         associated to the list

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2018-2023, CNRS
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
#include "calcephconfig.h"
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif

int main(void);

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    t_calcephbin *peph;

    char cname[CALCEPH_MAX_CONSTANTNAME];

    double values[4];

    int ret;

    int res = 0;

    int j;

    /* open the ephemeris file */
    peph = tests_calceph_open("checktpc_11627.tpc");
    ret = calceph_getconstant(peph, "BODY000_GMLIST1", values);
    if (ret != 1)
    {
        printf("find invalid value 'BODY000_GMLIST1' = %d\n", ret);
        res = 1;
    }

    ret = calceph_getconstant(peph, "BODY000_GMLIST2", values);
    if (ret != 2 || values[0] != 499.)
    {
        printf("find invalid value 'BODY000_GMLIST2' = %d\n", ret);
        res = 1;
    }

    ret = calceph_getconstant(peph, "BODY000_GMLIST4", values);
    if (ret != 4 || values[0] != 199.)
    {
        printf("find invalid value 'BODY000_GMLIST4' = %d\n", ret);
        res = 1;
    }

    for (j = 1; j <= calceph_getconstantcount(peph); j++)
    {
        ret = calceph_getconstantindex(peph, j, cname, values);
        if (strcmp(cname, "BODY000_GMLIST4") == 0 && ret != 4)
        {
            printf("find invalid value 'BODY000_GMLIST4' for calceph_getconstantindex = %d\n", ret);
            res = 1;
        }
        if (strcmp(cname, "BODY000_GMLIST2") == 0 && ret != 2)
        {
            printf("find invalid value 'BODY000_GMLIST2' for calceph_getconstantindex = %d\n", ret);
            res = 1;
        }
        if (strcmp(cname, "BODY000_GMLIST1") == 0 && ret != 1)
        {
            printf("find invalid value 'BODY000_GMLIST1' for calceph_getconstantindex = %d\n", ret);
            res = 1;
        }
    }

    calceph_close(peph);

    /* open the ephemeris file */
    peph = tests_calceph_open("matrix3.tf");
    ret = calceph_getconstant(peph, "TKFRAME_9900300_MATRIX", values);
    if (ret != 9 && values[0]!=0.)
    {
        printf("find invalid value 'TKFRAME_9900300_MATRIX' = %d\n", ret);
        res = 1;
    }
    for (j = 1; j <= calceph_getconstantcount(peph); j++)
    {
        ret = calceph_getconstantindex(peph, j, cname, values);
        if (strcmp(cname, "TKFRAME_9900300_MATRIX") == 0 && ret != 9)
        {
            printf("find invalid value 'TKFRAME_9900300_MATRIX' for calceph_getconstantindex = %d\n", ret);
            res = 1;
        }
    }
    calceph_close(peph);

    /* open the ephemeris file */
    tests_calceph_sopen("checktpc_11627.tpc");
    ret = calceph_sgetconstant("BODY000_GMLIST1", values);
    if (ret != 1)
    {
        printf("find invalid value 'BODY000_GMLIST1' = %d\n", ret);
        res = 1;
    }

    ret = calceph_sgetconstant("BODY000_GMLIST2", values);
    if (ret != 2 || values[0] != 499.)
    {
        printf("find invalid value 'BODY000_GMLIST2' = %d\n", ret);
        res = 1;
    }

    ret = calceph_sgetconstant("BODY000_GMLIST4", values);
    if (ret != 4 || values[0] != 199.)
    {
        printf("find invalid value 'BODY000_GMLIST4' = %d\n", ret);
        res = 1;
    }
    calceph_sclose();

    return res;
}
