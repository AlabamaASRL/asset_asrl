/*-----------------------------------------------------------------*/
/*! 
  \file calcephlocale.c 
  \brief locale independent data for strtod

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 

   Copyright, 2026, CNRS
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
#if HAVE_MATH_H
#include <math.h>
#endif
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include "calcephdebug.h"
#include "real.h"
#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"
#include "calcephinternal.h"
#include "util.h"

/*--------------------------------------------------------------------------*/
/*!
    Create the decimal dot locale for strtod
     return 0 on success
  @param clocale (inout) C locale.
*/
/*--------------------------------------------------------------------------*/
int calceph_locale_init(struct calceph_locale *clocale)
{
    int res = 0;

    clocale->useclocale = 0;
#if USE_STRTOD_L
#if HAVE__CREATE_LOCALE
    clocale->dataclocale = _create_locale(LC_NUMERIC, "C");
#else
    clocale->dataclocale = newlocale(LC_NUMERIC_MASK, "C", (locale_t) 0);
#endif
    clocale->useclocale = (clocale->dataclocale == (locale_t) 0) ? 0 : 1;
#endif
    /* if the locale does not produce a correct decimal point, generates an error */
    if (clocale->useclocale == 0)
    {
        char buffer[10];

        calceph_snprintf(buffer, 10, "%0.1f", 0.5);
        if (buffer[1] != '.')
        {
            fatalerror("Current locale does not create the decimal point '.' and calceph can't create a C locale\n");
            res = 1;
        }
    }
    return res;
}

/*--------------------------------------------------------------------------*/
/*!
    Free the locale 
  @param clocale (inout) C locale.
*/
/*--------------------------------------------------------------------------*/
void calceph_locale_clear(struct calceph_locale *clocale)
{
#if USE_STRTOD_L
    if (clocale->useclocale == 1)
        freelocale(clocale->dataclocale);
#else
    (void)clocale;
#endif
}
