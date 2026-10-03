/*-----------------------------------------------------------------*/
/*! 
  \file  checktpc_incremental.c
  \brief Check if the incremental assignment works on checktpc_incremental.tpc.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2021-2024, CNRS
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
#include <string.h>
#include "calceph.h"
#include "openfiles.h"

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    t_calcephbin *peph;
    int nb_value0;
    int nb_value5;
    double dvalue[5];
    t_calcephcharvalue svalue[5];
    int j;
    const char *filenamespk[] = { "../examples/example1.tpc", "checktpc_incremental.tpc" };

    /* open the ephemeris file */
    peph = tests_calceph_open_array(2, filenamespk);
    if (peph)
    {

        nb_value0 = calceph_getconstantvs(peph, "NAIF_BODY_NAME", NULL, 0);
        nb_value5 = calceph_getconstantvs(peph, "NAIF_BODY_NAME", svalue, 5);

        if (nb_value0 != 5)
        {
            for (j = 0; j < nb_value5; j++)
            {
                printf("%d : %s\n", j, svalue[j]);
            }
            printf("NAIF_BODY_NAME should have 5 constants (return %d)\n", nb_value0);

            return 1;
        }

        if (strcmp(svalue[0], "1 CERES") != 0 || strcmp(svalue[4], "JUNO") != 0)
        {
            printf("content of the string array is invalid.\n");
            for (j = 0; j < nb_value5; j++)
            {
                printf("%d : %s\n", j, svalue[j]);
            }
            return 1;
        }

        nb_value0 = calceph_getconstantvd(peph, "NAIF_BODY_CODE", NULL, 0);
        nb_value5 = calceph_getconstantvd(peph, "NAIF_BODY_CODE", dvalue, 5);
        if (nb_value0 != 5)
        {
            for (j = 0; j < nb_value5; j++)
            {
                printf("%d : %f\n", j, dvalue[j]);
            }
            printf("NAIF_BODY_CODE should have 5 constants (return %d)\n", nb_value0);

            return 1;
        }
        if (dvalue[0] != 2000001 || dvalue[4] != 2000003)
        {
            for (j = 0; j < nb_value5; j++)
            {
                printf("%d : %f\n", j, dvalue[j]);
            }
            printf("content of the floating-point array is invalid.\n");
            return 1;
        }

        calceph_close(peph);
    }
    else
        return 1;
    return 0;
}
