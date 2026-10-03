/*-----------------------------------------------------------------*/
/*! 
  \file corient.c 
  \brief Example of usage of the orientation functions with the python interface.
         This example computes for a date
          the angular momentum of Earth due to its rotation.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright 2017, CNRS
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

static void printcoord(double PV[6], const char *name);

int main(void);

/*-----------------------------------------------------------------*/
/* print coordinates */
/*-----------------------------------------------------------------*/
static void printcoord(double PV[6], const char *name)
{
    int j;

    printf("%s :\n", name);
    for (j = 0; j < 6; j++)
        printf("\t%23.16E\n", PV[j]);
    printf("\n");
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    double jd0 = 2451487;

    double dt = 0.5E0;

    double PV[6];

    t_calcephbin *peph;

    /* open the ephemeris file */
    peph = calceph_open("example2_rotangmom.dat");
    if (peph)
    {
        printf("The ephemeris is already opened\n");

        /* compute and print the coordinates */
        /* the angular momentum of Earth divided by (its mass times the square of its radius) due to its rotation */
        calceph_rotangmom_unit(peph, jd0, dt, NAIFID_EARTH, CALCEPH_UNIT_DAY + CALCEPH_USE_NAIFID, PV);
        printcoord(PV, "the angular momentum of Earth divided by (its mass times the square of its radius) due to its rotation in day^-1 : G/(mR^2)");

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
