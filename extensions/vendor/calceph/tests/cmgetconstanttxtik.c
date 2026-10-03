/*-----------------------------------------------------------------*/
/*!
  \file cmgetconstanttxtik.c
  \brief Check that calceph_getconstant and calceph_getconstantss return the number
         of values associated to the list for IK

  \author  D. De Araujo, M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2025, CNRS
   email of the author : Mickael.Gastineau@obspm.fr
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
knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its
terms.
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
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    t_calcephbin *peph;

    t_calcephcharvalue svalue;

    double values[3];

    int ret;

    int res = 0;

    /* open the ephemeris file */
    peph = tests_calceph_open("example_ik.ti");

    ret = calceph_getconstantss(peph, "INS-42550_FOV_CLASS_SPEC", svalue);
    if (ret != 1 || strncmp("ANGLES", svalue, 6) != 0)
    {
        printf("find invalid value 'INS-42550_FOV_CLASS_SPEC' = %s(ret = %d)\n", svalue, ret);
        res = 1;
    }

    ret = calceph_getconstantss(peph, "INS-42550_FOV_SHAPE", svalue);
    if (ret != 1 || strncmp("CIRCLE", svalue, 6) != 0)
    {
        printf("find invalid value 'INS-42550_FOV_SHAPE' = %s(ret = %d)\n", svalue, ret);
        res = 1;
    }

    ret = calceph_getconstantss(peph, "INS-42550_FOV_FRAME", svalue);
    if (ret != 1 || strncmp("EXAMPLE_CIRCLE", svalue, 14) != 0)
    {
        printf("find invalid value 'INS-42550_FOV_FRAME' = %s (ret = %d)\n", svalue, ret);
        res = 1;
    }

    ret = calceph_getconstant(peph, "INS-42550_BORESIGHT", values);
    if (ret != 3 || values[0] != 0.)
    {
        printf("find invalid value 'INS-42550_BORESIGHT' = %f (ret = %d)\n", values[0], ret);
        res = 1;
    }

    ret = calceph_getconstant(peph, "INS-42550_FOV_REF_VECTOR", values);
    if (ret != 3 || values[0] != -1.)
    {
        printf("find invalid value 'INS-42550_FOV_REF_VECTOR' = %f (ret = %d)\n", values[0], ret);
        res = 1;
    }

    ret = calceph_getconstant(peph, "INS-42550_FOV_REF_ANGLE", values);
    if (ret != 1 || values[0] != 7.)
    {
        printf("find invalid value 'INS-42550_FOV_REF_ANGLE' = %f (ret = %d)\n", values[0], ret);
        res = 1;
    }

    ret = calceph_getconstantss(peph, "INS-42550_FOV_ANGLE_UNITS", svalue);
    if (ret != 1 || strncmp("DEGREES", svalue, 6) != 0)
    {
        printf("find invalid value 'INS-42550_FOV_ANGLE_UNITS' = %s (ret = %d)\n", svalue, ret);
        res = 1;
    }

    calceph_close(peph);

    return res;
}
