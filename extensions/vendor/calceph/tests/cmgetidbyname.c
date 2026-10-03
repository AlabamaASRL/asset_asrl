/*-----------------------------------------------------------------*/
/*! 
  \file cmgetidbyname.c 
  \brief Check that calceph_getidbyname retuns the number of values 
         associated to the list

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2023, CNRS
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
/* check the invalid name */
/*-----------------------------------------------------------------*/
static int check_failure(t_calcephbin * peph, const char *name, int unit)
{
    int id = -1;

    int ret = calceph_getidbyname(peph, name, unit, &id);

    if (ret != 0)
    {
        printf("failure : find valid id but the name is invalid !!!\n");
        printf("          name='%s' unit=%d id=%d ret=%d\n", name, unit, id, ret);
    }
    return ret;
}

/*-----------------------------------------------------------------*/
/* check the valid name */
/*-----------------------------------------------------------------*/
static int check_valid(t_calcephbin * peph, const char *name, int unit, int id_expected)
{
    int id = -1;

    int ret = calceph_getidbyname(peph, name, unit, &id);

    if (ret != 1 || id_expected != id)
    {
        printf("failure : find invalid id but the name is valid !!!\n");
        printf("          name='%s' unit=%d id=%d id_expected=%d ret=%d\n", name, unit, id, id_expected, ret);
        ret = 1;
    }
    else ret = 0;
    return ret;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    t_calcephbin *peph;

    int res = 0;

    /* open the ephemeris file */
    peph = tests_calceph_open("../examples/example1.tpc");
    if (peph)
    {
        res += check_failure(peph, "", 0);
        res += check_failure(peph, "JUPITERBARYCENTER", 0);
        res += check_failure(peph, "", CALCEPH_USE_NAIFID);
        res += check_failure(peph, "JUPITERBARYCENTER", CALCEPH_USE_NAIFID);
        res += check_failure(peph, "XX", 0);
        res += check_failure(peph, "XX", CALCEPH_USE_NAIFID);

        res += check_valid(peph, "JUPITER BARYCENTER", 0, 5);
        res += check_valid(peph, "Jupiter Barycenter", 0, 5);
        res += check_valid(peph, "   Jupiter Barycenter", 0, 5);
        res += check_valid(peph, "Jupiter    Barycenter", 0, 5);
        res += check_valid(peph, "   Jupiter Barycenter   ", 0, 5);

        res += check_valid(peph, "SATURN BARYCENTER", CALCEPH_USE_NAIFID, 6);
        res += check_valid(peph, "SaTURN Barycenter", CALCEPH_USE_NAIFID, 6);
        res += check_valid(peph, "   Saturn Barycenter", CALCEPH_USE_NAIFID, 6);
        res += check_valid(peph, "Saturn    Barycenter", CALCEPH_USE_NAIFID, 6);
        res += check_valid(peph, "   Saturn Barycenter   ", CALCEPH_USE_NAIFID, 6);

        res += check_valid(peph, "SATURN ", CALCEPH_USE_NAIFID, 699);
        res += check_valid(peph, "SaTURN    ", CALCEPH_USE_NAIFID, 699);
        res += check_valid(peph, "   Saturn", CALCEPH_USE_NAIFID, 699);
        res += check_valid(peph, "Saturn", CALCEPH_USE_NAIFID, 699);
        res += check_valid(peph, "   Saturn   ", CALCEPH_USE_NAIFID, 699);

        res += check_valid(peph, "Mercury Barycenter", 0, 1);
        res += check_valid(peph, "Venus Barycenter", 0, 2);
        res += check_valid(peph, " Earth ", 0, 3);
        res += check_valid(peph, "Mars Barycenter", 0, 4);
        res += check_valid(peph, "Saturn Barycenter", 0, 6);
        res += check_valid(peph, "Uranus Barycenter", 0, 7);
        res += check_valid(peph, "Neptune Barycenter", 0, 8);
        res += check_valid(peph, "Pluto Barycenter", 0, 9);
        res += check_valid(peph, "Moon", 0, 10);
        res += check_valid(peph, "Sun", 0, 11);
        res += check_valid(peph, "Solar   System   barycenter", 0, 12);
        res += check_valid(peph, " Earth Moon   barycenter ", 0, 13);

        res += check_valid(peph, " Earth ", CALCEPH_USE_NAIFID, 399);
        res += check_valid(peph, " Moon ", CALCEPH_USE_NAIFID, 301);
        res += check_valid(peph, "Sun", CALCEPH_USE_NAIFID, 10);
        res += check_valid(peph, "Solar   System   barycenter", CALCEPH_USE_NAIFID, 0);
        res += check_valid(peph, " Earth Moon   barycenter ", CALCEPH_USE_NAIFID, 3);

        res += check_valid(peph, "2 PALLAS", CALCEPH_USE_NAIFID, 2000002);
        res += check_valid(peph, "1 CERES", CALCEPH_USE_NAIFID, 2000001);


        calceph_close(peph);
    }
    else
        res = 1;

    return res;
}
