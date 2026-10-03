/*-----------------------------------------------------------------*/
/*! 
  \file corder.c 
  \brief Example of usage of the caclceph_compute_order functions.
         It computes for a date
          the geocentric moon coordinates, 
          the heliocentric quantities of Mars.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2016, 2017, CNRS
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

static void printcoord(int order, const double *PVAJ, const char *name);

int main(void);

/*-----------------------------------------------------------------*/
/* print coordinates */
/*-----------------------------------------------------------------*/
static void printcoord(int order, const double *PVAJ, const char *name)
{
    int j;

    printf("%s : \norder=%d\n", name, order);
    printf("Position     :");
    for (j = 0; j < 3; j++)
        printf(" %23.16E", PVAJ[j]);
    printf("\n");
    if (order >= 1)
    {
        printf("Velocity     :");
        for (j = 0; j < 3; j++)
            printf(" %23.16E", PVAJ[j + 3]);
        printf("\n");
    }
    if (order >= 2)
    {
        printf("Acceleration :");
        for (j = 0; j < 3; j++)
            printf(" %23.16E", PVAJ[j + 6]);
        printf("\n");
    }
    if (order >= 3)
    {
        printf("Jerk         :");
        for (j = 0; j < 3; j++)
            printf(" %23.16E", PVAJ[j + 9]);
        printf("\n");
    }
    printf("\n");
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    double jd0 = 2442457;

    double dt = 0.5E0;

    double PVAJ[12];

    int order;

    t_calcephbin *peph;

    /* open the ephemeris file */
    peph = calceph_open("example1.dat");
    if (peph)
    {
        printf("The ephemeris is already opened\n");

        /* compute and print the coordinates */
        /* the geocentric moon coordinates */
        for (order = 0; order <= 3; order++)
        {
            calceph_compute_order(peph, jd0, dt, 10, 3, CALCEPH_UNIT_KM + CALCEPH_UNIT_SEC, order, PVAJ);
            printcoord(order, PVAJ, "geocentric coordinates of the Moon in km and seconds");
        }

        /* the heliocentric coordinates of Mars */
        for (order = 0; order <= 3; order++)
        {
            calceph_compute_order(peph, jd0, dt, 4, 11, CALCEPH_UNIT_KM + CALCEPH_UNIT_SEC, order, PVAJ);
            printcoord(order, PVAJ, "heliocentric coordinates of Mars");
        }

        /* close the ephemeris file */
        calceph_close(peph);
        printf("The ephemeris is already closed\n");
    }
    else
    {
        printf("The ephemeris can't be opened\n");
    }
    return 0;
}
