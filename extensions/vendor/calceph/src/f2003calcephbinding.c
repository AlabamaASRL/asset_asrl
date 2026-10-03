/*-----------------------------------------------------------------*/
/*!
  \file f2003calcephbinding.c
  \brief Fortran 2003 interface for Calceph : C binding.

  \author  M. Gastineau, H. Manche
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright, 2008-2026,CNRS
   email of the author : Mickael.Gastineau@obspm.fr, Herve.Manche@obspm.fr

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

#include "calcephconfig.h"
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif
#if HAVE_ERRNO_H
#include <errno.h>
#endif
#include "real.h"
#include "util.h"
#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"
#include "calcephinternal.h"

#if __STRICT_ANSI__
#define inline
#endif

#include "f90fillspace.h"
#include "f2003calcephbinding.h"

/********************************************************/
/*! Open the list of specified ephemeris file  - Fortran 2003 interface
  Use C binding interface.
  return the ephemeris descriptor.

 @param n (in)  number of file
 @param filename (in) array of file name
 @param len (in) length of each file name
*/
/********************************************************/
void *f2003calceph_open_array(int n, char *filename, int len)
{
    int k, l;

    char **filear;

    char *newname;

    void *eph = NULL;

    filear = (char **) malloc(sizeof(char *) * (n));
    newname = (char *) malloc(sizeof(char) * (len + 1) * (n));
    if (newname != NULL && filear != NULL)
    {
        for (k = 0; k < n; k++)
        {
            filear[k] = newname + (len + 1) * k;
            memcpy(filear[k], filename + len * k, len * sizeof(char));
            filear[k][len] = '\0';
            l = len - 1;
            while (l > 0 && filear[k][l] == ' ')
            {
                filear[k][l] = '\0';
                l--;
            }
        }
        eph = calceph_open_array(n, (const char *const *) filear);
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for f90calceph_open\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */
    if (newname != NULL)
        free(newname);
    if (filear != NULL)
        free(filear);
    return eph;
}

/********************************************************/
/*! get the name and its value of the constant available
    at some index from the ephemeris file
   return 1 if index is valid
   return 0 if index isn't valid

  @param index (in) index of the constant (must be between 1 and
  calceph_sgetconstantcount() )
  @param name (out) name of the constant
  @param value (out) value of the constant
*/
/********************************************************/
int f2003calceph_sgetconstantindex(int index, char name[CALCEPH_MAX_CONSTANTNAME], double *value)
{
    int res = calceph_sgetconstantindex(index, name, value);

    if (res == 1)
    {
        calceph_fortranfillspace(name, CALCEPH_MAX_CONSTANTNAME);
    }
    return res;
}

/********************************************************/
/*! store, in version, the file version of the ephemeris file.
   return 0 if the file version was not found.
   return 1 on sucess.

  @param version (out) fortran string of the version of the ephemeris
  file

*/
/********************************************************/
int f2003calceph_sgetfileversion(char version[CALCEPH_MAX_CONSTANTVALUE])
{
    int res = calceph_sgetfileversion(version);

    if (res == 1)
        calceph_fortranfillspace(version, CALCEPH_MAX_CONSTANTVALUE);
    return res;
}

/********************************************************/
/*! return the name and the associated value of the constant available
    at some index from the ephemeris file
   return 1 if index is valid
   return 0 if index isn't valid

  @param eph (inout) ephemeris descriptor
  @param index (in) index of the constant (must be between 1 and
  calceph_getconstantcount() )
  @param name (out) name of the constant
  @param value (out) value of the constant
*/
/********************************************************/
int f2003calceph_getconstantindex(t_calcephbin *eph, int index, char name[CALCEPH_MAX_CONSTANTNAME], double *value)
{
    int k;

    int res = calceph_getconstantindex(eph, index, name, value);

    if (res == 1)
    {
        for (k = 0; k < CALCEPH_MAX_CONSTANTNAME - 1; k++)
        {
            if (name[k] == '\0')
                name[k] = ' ';
        }
    }
    return res;
}

/********************************************************/
/*! store, in version, the file version of the ephemeris file.
   return 0 if the file version was not found.
   return 1 on sucess.

  @param eph (inout) ephemeris descriptor
  @param szversion (out) fortran string of the version of the ephemeris
  file

*/
/********************************************************/
int f2003calceph_getfileversion(t_calcephbin *eph, char version[CALCEPH_MAX_CONSTANTVALUE])
{
    int res = calceph_getfileversion(eph, version);

    if (res == 1)
        calceph_fortranfillspace(version, CALCEPH_MAX_CONSTANTVALUE);
    return res;
}

/********************************************************/
/*! return the version as a string.
  The trailing character are filled with a space character.

  @param version (out) name of the version
*/
/********************************************************/
void f2003calceph_getversion_str(char version[CALCEPH_MAX_CONSTANTNAME])
{
    calceph_getversion_str(version);
    calceph_fortranfillspace(version, CALCEPH_MAX_CONSTANTNAME);
}
