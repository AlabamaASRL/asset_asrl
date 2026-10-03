/*-----------------------------------------------------------------*/
/*! 
  \file cgetorientrecordindex.c
  \brief Check if calceph_getorientrecordindex(2) works on INPOP files

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright,  2009-2021, CNRS
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

int main(void);

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
static int maincheck(const char *file, int maxvalue, int segid_expected)
{
    t_calcephbin *peph;

    int res = 0;

    int j;

    int countorientrecord;

    double firsttime, lasttime;

    int target, frame, segid;

    /* open the ephemeris file */
    peph = tests_calceph_open(file);
    if (peph)
    {
        res = 1;
        countorientrecord = calceph_getorientrecordcount(peph);
        for (j = 1; j <= countorientrecord && res == 1; j++)
        {
            res = calceph_getorientrecordindex(peph, j, &target,  &firsttime, &lasttime, &frame);
            if (calceph_getorientrecordindex2(peph, j, &target,  &firsttime, &lasttime, &frame, &segid) != 1)
            {
                printf("failure calceph_getorientrecordindex2 index= %d\n", j);
                res = 0;
            }
            if (segid_expected!=segid)
            {
                printf("failure calceph_getorientrecordindex2 at index=%d expected= %d computed=%d\n", j, segid_expected, segid);
                res = 0;
            }
        }
        if (countorientrecord <= 0 || countorientrecord > maxvalue)
        {
            printf("failure countorientrecord bad value= %d\n", countorientrecord);
            res = 0;
        }
        if (calceph_getorientrecordindex(peph, 0, &target, &firsttime, &lasttime, &frame) != 0)
        {
            printf("failure calceph_getorientrecordindex index= 0\n");
            res = 0;
        }
        if (calceph_getorientrecordindex
            (peph, countorientrecord + 1, &target,  &firsttime, &lasttime, &frame) != 0)
        {
            printf("failure calceph_getorientrecordindex index= %d\n", countorientrecord + 1);
            res = 0;
        }
        calceph_close(peph);
    }
    return (res == 1 ? 0 : 1);
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    if (maincheck("../examples/example1.dat", 1, CALCEPH_SEGTYPE_ORIG_0)!=0) return 1;
    return maincheck("../examples/example1.bpc", 1, CALCEPH_SEGTYPE_SPK_2);
}
