/*-----------------------------------------------------------------*/
/*! 
  \file openfiles.c 
  \brief Open files for the tests.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 

   Copyright, 2015-2024, CCNRS
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
#include <stdlib.h>
#if HAVE_STRING_H
#include <string.h>
#endif
#include "calceph.h"
#include "openfiles.h"
#include "real.h"
#include "calcephinternal.h"
#include "util.h"

static char *prefixfilename(const char *filename);

/*-----------------------------------------------------------------*/
/*! return the filename prefixed by the srcdir
    memory allocated must be freed using free
   @param filename (in) file to be opened
*/
/*-----------------------------------------------------------------*/
static char *prefixfilename(const char *filename)
{
    char *pathname;
    char defaultdir[] = ".";
    char *srcdir;
    size_t lenpath;

    srcdir = getenv("srcdir");
    if (srcdir == NULL)
        srcdir = defaultdir;

    lenpath = strlen(srcdir) + strlen(filename) + 2;
    pathname = (char *) malloc(sizeof(char) * lenpath);
    if (pathname == NULL)
    {
        printf("can't allocate memory\n");
        return NULL;
    }
    calceph_snprintf(pathname, lenpath, "%s/%s", srcdir, filename);
    return pathname;
}

/*-----------------------------------------------------------------*/
/*! prefix filename using srcdir to open the files using fopen 
   @param filename (in) file to be opened
   @param mode (in) mode used for fopen
*/
/*-----------------------------------------------------------------*/
FILE *tests_fopen(const char *filename, const char *mode)
{
    FILE *f;
    char *fullfilename = prefixfilename(filename);

    if (fullfilename == NULL)
        return NULL;
    f = fopen(fullfilename, mode);
    free(fullfilename);
    return f;
}

/*-----------------------------------------------------------------*/
/*! prefix filename using srcdir to open the files using calceph_open 
   @param filename (in) file to be opened
*/
/*-----------------------------------------------------------------*/
t_calcephbin *tests_calceph_open(const char *filename)
{
    t_calcephbin *f;
    char *fullfilename = prefixfilename(filename);

    if (fullfilename == NULL)
        return NULL;
    f = calceph_open(fullfilename);
    free(fullfilename);
    return f;
}

/*-----------------------------------------------------------------*/
/*! prefix filename using srcdir to open the files using 
 calceph_open_array
   @param n (in) number of elements in the array filename
   @param filename (in) array of files to be opened
*/
/*-----------------------------------------------------------------*/
t_calcephbin *tests_calceph_open_array(int n, const char *const filename[ /*n */ ])
{
    int k;
    t_calcephbin *f;
    char **fullfilename;

    fullfilename = (char **) malloc(sizeof(char *) * n);
    if (fullfilename == NULL)
        return NULL;
    for (k = 0; k < n; k++)
    {
        fullfilename[k] = prefixfilename(filename[k]);
        if (fullfilename[k] == NULL)
            return NULL;
    }
    f = calceph_open_array2(n, fullfilename);
    for (k = 0; k < n; k++)
    {
        free(fullfilename[k]);
    }
    free(fullfilename);

    return f;
}

/*-----------------------------------------------------------------*/
/*! prefix filename using src_dir to open the files using calceph_sopen 
   @param filename (in) file to be opened
*/
/*-----------------------------------------------------------------*/
int tests_calceph_sopen(const char *filename)
{
    int f;
    char *fullfilename = prefixfilename(filename);

    if (fullfilename == NULL)
        return 0;
    f = calceph_sopen(fullfilename);
    free(fullfilename);
    return f;
}

/*-----------------------------------------------------------------*/
/*! prefix filename using srcdir to open csv files using fopen 
   @param filename (in) file to be opened
*/
/*-----------------------------------------------------------------*/
FILE *tests_calceph_open_r(const char *filename)
{
    FILE *f;
    char *fullfilename = prefixfilename(filename);

    if (fullfilename == NULL)
        return NULL;
    f = fopen(fullfilename, "r");
    free(fullfilename);
    return f;
}
