/*-----------------------------------------------------------------*/
/*!
  \file cmgetnamebyidss.c
  \brief Check that calceph_getnamebyidss retuns the number of values
         associated to the list

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris.

   Copyright, 2024, CNRS
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
static int check_failure(t_calcephbin* peph, int id, int unit)
{
  char name[CALCEPH_MAX_CONSTANTVALUE];

  int ret = calceph_getnamebyidss(peph, id, unit, name);

  if (ret != 0)
  {
    printf("failure : find valid name but the id is invalid !!!\n");
    printf("          name='%s' unit=%d id=%d ret=%d\n", name, unit, id, ret);
  }
  return ret;
}

/*-----------------------------------------------------------------*/
/* check the valid name */
/*-----------------------------------------------------------------*/
static int check_valid(t_calcephbin* peph, int id, int unit, const char* name_expected)
{
  char name[CALCEPH_MAX_CONSTANTVALUE];

  int ret = calceph_getnamebyidss(peph, id, unit, name);

  if (ret != 1 || strcmp(name, name_expected) != 0)
  {
    printf("failure : find invalid name but the id is valid !!!\n");
    printf("          id=%d unit=%d name='%s' name_expected='%s' ret=%d\n", id, unit, name,
           name_expected, ret);
    ret = 1;
  }
  else
    ret = 0;
  return ret;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
  t_calcephbin* peph;

  int res = 0;

  /* open the ephemeris file */
  peph = tests_calceph_open("../examples/example1.tpc");
  if (peph)
  {
    res += check_failure(peph, -5000, 0);
    res += check_failure(peph, 998, 0);
    res += check_failure(peph, 21, 0);
    res += check_failure(peph, -5000, CALCEPH_USE_NAIFID);
    res += check_failure(peph, 998, CALCEPH_USE_NAIFID);

    res += check_valid(peph, 5, 0, "JUPITER BARYCENTER");

    res += check_valid(peph, 6, CALCEPH_USE_NAIFID, "SATURN BARYCENTER");

    res += check_valid(peph, 699, CALCEPH_USE_NAIFID, "SATURN");

    res += check_valid(peph, 1, 0, "MERCURY BARYCENTER");
    res += check_valid(peph, 2, 0, "VENUS BARYCENTER");
    res += check_valid(peph, 3, 0, "EARTH");
    res += check_valid(peph, 4, 0, "MARS BARYCENTER");
    res += check_valid(peph, 6, 0, "SATURN BARYCENTER");
    res += check_valid(peph, 7, 0, "URANUS BARYCENTER");
    res += check_valid(peph, 8, 0, "NEPTUNE BARYCENTER");
    res += check_valid(peph, 9, 0, "PLUTO BARYCENTER");
    res += check_valid(peph, 10, 0, "MOON");
    res += check_valid(peph, 11, 0, "SUN");
    res += check_valid(peph, 12, 0, "SOLAR SYSTEM BARYCENTER");
    res += check_valid(peph, 13, 0, "EARTH MOON BARYCENTER");

    res += check_valid(peph, 4, CALCEPH_USE_NAIFID, "MARS BARYCENTER");
    res += check_valid(peph, 499, CALCEPH_USE_NAIFID, "MARS");
    res += check_valid(peph, 399, CALCEPH_USE_NAIFID, "EARTH");
    res += check_valid(peph, 301, CALCEPH_USE_NAIFID, "MOON");
    res += check_valid(peph, 10, CALCEPH_USE_NAIFID, "SUN");
    res += check_valid(peph, 0, CALCEPH_USE_NAIFID, "SOLAR SYSTEM BARYCENTER");
    res += check_valid(peph, 3, CALCEPH_USE_NAIFID, "EARTH MOON BARYCENTER");

    res += check_valid(peph, 2000002, CALCEPH_USE_NAIFID, "2 PALLAS");
    res += check_valid(peph, 2000001, CALCEPH_USE_NAIFID, "1 CERES");


    calceph_close(peph);
  }
  else
    res = 1;

  return res;
}
