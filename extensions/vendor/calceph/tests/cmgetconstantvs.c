/*-----------------------------------------------------------------*/
/*! 
  \file cmgetconstant_vs.c 
  \brief Check that calceph_getconstantvs returns the number of values 
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

    char values[4][CALCEPH_MAX_CONSTANTVALUE];

    int ret;

    int res = 0;

    /* open the ephemeris file */
    peph = tests_calceph_open("checktpc_str.tpc");
    
    ret = calceph_getconstantss(peph, "DISTANCE_UNITS", values[0]);
    if (ret != 1 || strcmp(values[0],"KILOMETERS")!=0)
    {
        printf("find invalid value 'DISTANCE_UNITS' = '%s'\n", values[0]);
        res = 1;
    }
    ret = calceph_getconstantvs(peph, "DISTANCE_UNITS", values, 4);
    if (ret != 1|| strcmp(values[0],"KILOMETERS")!=0)
    {
        printf("find invalid value 'DISTANCE_UNITS' = '%s'\n", values[0]);
        res = 1;
    }

    ret = calceph_getconstantss(peph, "MISSION_UNITS", values[0]);
    if (ret != 3 || strcmp(values[0],"KILOMETERS")!=0)
    {
        printf("find invalid value 'MISSION_UNITS' = '%s'\n", values[0]);
        res = 1;
    }
    ret = calceph_getconstantvs(peph, "MISSION_UNITS", values, 4);
    if (ret != 3 || strcmp(values[0],"KILOMETERS")!=0 || strcmp(values[1],"SECONDS")!=0 || strcmp(values[2],"KILOMETERS/SECOND")!=0)
    {
        printf("find invalid value 'MISSION_UNITS' = '%s' '%s' '%s'\n", values[0], values[1], values[2]);
        res = 1;
    }

    ret = calceph_getconstantss(peph, "MESSAGE", values[0]);
    if (ret != 1 || strcmp(values[0],"You can't always get what you want.")!=0)
    {
        printf("find invalid value 'MESSAGE' = '%s'\n", values[0]);
        res = 1;
    }
    ret = calceph_getconstantvs(peph, "MESSAGE", values, 4);
    if (ret != 1 || strcmp(values[0],"You can't always get what you want.")!=0)
    {
        printf("find invalid value 'MESSAGE' = '%s'\n", values[0]);
        res = 1;
    }

    calceph_close(peph);


    return res;
}
