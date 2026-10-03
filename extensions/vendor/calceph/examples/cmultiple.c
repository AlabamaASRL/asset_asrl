/*-----------------------------------------------------------------*/
/*! 
  \file cmultiple.c 
  \brief Example of usage of the multiple file access functions.
         This example reads the constant EMRAT, AU, GM_Mer and print their values.
         It computes for a date
          the geocentric moon coordinates, 
          the value TT-TDB 
          the heliocentric coordinates of Mars.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2008-2023, CNRS
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

static void printcoord(double PV[6],const char *name);
int main(void);

/*-----------------------------------------------------------------*/
/* print coordinates */
/*-----------------------------------------------------------------*/
static void printcoord(double PV[6],const char *name)
{
 int j;
 printf("%s :\n", name);
 for (j=0; j<6; j++) printf("\t%23.16E\n", PV[j]);
 printf("\n");
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
 double AU, EMRAT, GM_Mer;
 double jd0=2442457;
 double dt=0.5E0;
 double PV[6];
 int j;
 t_calcephbin *peph;
 double valueconstant;
 char nameconstant[CALCEPH_MAX_CONSTANTNAME];
 int earth, moon;
 
 /* open the ephemeris file */
 peph = calceph_open("example1.dat");
 if (peph)
 {
    printf("The ephemeris is already opened\n");
   /* print the values of AU, EMRAT and GM_Mer */
   if (calceph_getconstant(peph, "AU", &AU)) printf("AU=%23.16E\n", AU);
   if (calceph_getconstant(peph, "EMRAT", &EMRAT)) printf("EMRAT=%23.16E\n", EMRAT);
   if (calceph_getconstant(peph, "GM_Mer", &GM_Mer)) printf("GM_Mer=%23.16E\n", GM_Mer);
 
   /* compute and print the coordinates */
   /* the geocentric moon coordinates */
   calceph_getidbyname(peph, "Earth",0, &earth);
   calceph_getidbyname(peph, "Moon",0, &moon);
   calceph_compute(peph, jd0, dt, moon, earth, PV);
   printcoord(PV,"geocentric coordinates of the Moon in AU and AU/day");
   calceph_compute_unit(peph, jd0, dt, moon, earth, CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC, PV);
   printcoord(PV,"geocentric coordinates of the Moon in km and km/s");
   
   /* the value TT-TDB */
   if (calceph_compute(peph, jd0, dt, 16, 0, PV))
   {
      printf("TT-TDB = %23.16E\n", PV[0]);
   }
   printf("mars\n");
   /* the heliocentric coordinates of Mars */
   calceph_compute(peph, jd0, dt, 4, 11, PV);
   printcoord(PV,"heliocentric coordinates of Mars");
 
 
   /* print the whole list of the constants */
   printf("list of constants\n");
   for (j=1; j<=calceph_getconstantcount(peph); j++)
   {
    calceph_getconstantindex(peph, j, nameconstant, &valueconstant);
    printf("'%s'\t= %23.16E\n", nameconstant, valueconstant);
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
