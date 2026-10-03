/*-----------------------------------------------------------------*/
/*! 
  \file cparallel.c 
  \brief Example of usage of the parallel access to the same object.
         This example computes the mean distance between earth and moon.
         You can adjust the number of threads using the enviromental 
         variable OMP_NUM_THREADS.
         This example should be the openmp flags of your compiler, 
         such a  -fopenmp or -qopenmp


  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2019, CNRS
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
#include <math.h>
#include "calceph.h"

int main(void);


/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{

 double jd0 = 2442458;
 double dt = 0.001;;
 int  NPOINTS = 1000000;
 double d, sum;
 int j;
 t_calcephbin *peph;
 int threadsafe;
 
 /* open the ephemeris file */
 peph = calceph_open("example1.dat");
 if (peph)
 {
    printf("The ephemeris is already opened\n");
    
    /* prefetch the data: it is required to enable later  parallel access */
    calceph_prefetch(peph);
    threadsafe = calceph_isthreadsafe(peph);
    if (threadsafe!=0) 
    {
    printf("The multiple access to the ephemeris descriptor is thread-safe.\n");
    printf("The loop 'for' will be parallelized.\n");
    printf("The number of used threads is determined by OMP_NUM_THREADS\n");
    }
    else
    {
    printf("The multiple access to the ephemeris descriptor is NOT thread-safe.\n");
    printf("The loop 'for' will be executed in sequential.\n");
    }
    
 
    /* parallel loop */
#pragma omp parallel for reduction (+:sum) if (threadsafe!=0)
    for (j=0; j<NPOINTS; j++)
    {
       double PV[6]; 
       calceph_compute_unit(peph, jd0, j*dt, NAIFID_MOON, NAIFID_EARTH, CALCEPH_UNIT_KM+CALCEPH_UNIT_SEC+CALCEPH_USE_NAIFID,  PV);
       sum = sum + sqrt(PV[0]*PV[0]+PV[1]*PV[1]+PV[2]*PV[2]);
    }
 
    d = sum/NPOINTS;
    printf("mean distance between the Earth and the Moon : %.3f Kilometers\n", d);
       
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
