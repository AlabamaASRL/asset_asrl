/*-----------------------------------------------------------------*/
/*! 
  \file cscompute.c 
  \brief simple computation with a file and a date provided on the command line
   e.g.: cscompute  myfile.dat 2.451545E+06
   or .e.g:  cscompute  myfile.bsp myfile.tpc  2.451545E+06

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2009-2023, CNRS
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
#include <stdlib.h>
#include "calceph.h"

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(int argc, const char * const argv[])
{
 int res = 0;
 t_calcephbin *peph;
 double jd0;
 double PV[6];
 int earth, sun;
 
 if (argc<3)
 {
  fprintf(stderr, "Usage : cscompute filename1 filename2 ... filenamen juliandate\n");
  fprintf(stderr, "                  filename? : binary or spice ephemeris file names\n");
  fprintf(stderr, "                  juliandate : julian date (floating-point number)\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "e.g.  : cscompute inpop08a_m100_p100_littleendian.dat 2.451545E+06\n");
  fprintf(stderr, "e.g.  : cscompute de421.bsp masses.tpc 2.451545E+06\n");
  return 0;
 }
 
 /* open the ephemeris file */
 peph = calceph_open_array(argc-2, argv+1);
 if (peph!=NULL)
 {
       jd0 =atof(argv[argc-1]);
       printf("julian date :%e\n", jd0);
       
       /* compute  the coordinates */
       calceph_getidbyname(peph, "Earth",0, &earth);
       calceph_getidbyname(peph, "Sun",0, &sun);
       res = calceph_compute(peph, jd0, 0.E0, earth, sun, PV);
       if (res==1)
       {
        printf("cartesian heliocentric position of the Earth : %g %g %g\n", PV[0], PV[1], PV[2]);
        printf("cartesian heliocentric velocity of the Earth : %g %g %g\n", PV[3], PV[4], PV[5]);
       }
       
       /* close the ephemeris file */
       calceph_close(peph);
 }
 return res;
}
