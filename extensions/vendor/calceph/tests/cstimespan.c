/*-----------------------------------------------------------------*/
/*! 
  \file cstimespan.c 
  \brief Check if calceph_sgettimespan works

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2018, CNRS
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

static double myabs(double x);

int main(void);

static double myabs(double x)
{
    if (x!=x) return 1E300;
    return (x > 0 ? x : -x);
}

/*-----------------------------------------------------------------*/
/* check function */
/*-----------------------------------------------------------------*/
static int maincheck(const char *filename, double firsttime_expected, double lasttime_expected, int countinuous_expected)
{
    double firsttime = 0.0, lasttime = 0.0;
    int countinuous = -1;

    int res = 0;

    /* open the ephemeris file */
    res = tests_calceph_sopen(filename);
    if (res) res = 0;
    if (calceph_sgettimespan(&firsttime, &lasttime, &countinuous)==0)
    {
        printf("calceph_sgettimespan fails '%s'\n",  filename);
        res = 1;
    }
    if (myabs(firsttime-firsttime_expected)>1E-13)
    {
        printf("calceph_sgettimespan fails for first time '%s' : computed=%23.16E expected=%23.16E \n",  filename, firsttime, firsttime_expected );
        res = 1;
    }
    if (myabs(lasttime-lasttime_expected)>1E-13)
    {
        printf("calceph_sgettimespan fails for last time '%s' : computed=%23.16E expected=%23.16E \n",  filename, lasttime, lasttime_expected );
        res = 1;
    }
    if (countinuous!=countinuous_expected)
    {
        printf("calceph_sgettimespan fails for countinuous '%s' : computed=%d expected=%d \n",  filename, countinuous, countinuous_expected );
        res = 1;
    }
    calceph_sclose();
    return res;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    int res;
    const char *filearinpop = "../examples/example1.dat";
    const char *filenamespk =  "../examples/example1.bsp";

    res = maincheck(filearinpop, 2442457.0, 2451545.0, 1);
    if (res == 0)
        res = maincheck(filenamespk, 2442457.0, 2451545.0, 1);
    return res;
}
