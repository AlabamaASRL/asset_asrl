/*-----------------------------------------------------------------*/
/*! 
  \file calcephthread.c 
  \brief portable thread functions.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 

   Copyright, 2006-2026, CNRS
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
#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"
#include "real.h"
#include "util.h"
#include "calcephinternal.h"
#include "calcephthread.h"

/* maximum number of threads*/
#define MAX_NTHREADS 10

int calceph_thread_start_and_join(int nthreads, thrd_start_t fns[], void *args[])
{
    int j;
    int ret = 0;

#if HAVE_ISO_C11_THREAD
    thrd_t thread[MAX_NTHREADS];

    for (j = 0; j < nthreads; j++)
    {
        if (thrd_create(thread + j, fns[j], args[j]) != thrd_success)
        {
            printf("thrd_create fails\n");
            exit(1);
        }
    }
    for (j = 0; j < nthreads; j++)
    {
        int res;

        thrd_join(thread[j], &res);
        if (res == thrd_error)
            ret = 1;
    }
#elif HAVE_PTHREAD
    pthread_t thread[MAX_NTHREADS];

    for (j = 0; j < nthreads; j++)
    {
        if (pthread_create(thread + j, NULL, fns[j], args[j]) != 0)
        {
            printf("pthread_create fails\n");
            exit(1);
        }
    }
    for (j = 0; j < nthreads; j++)
    {
        void *res;

        pthread_join(thread[j], &res);
        if (res != NULL)
            ret = 1;
    }
    return ret;

#elif HAVE_WIN32API
    HANDLE hthread[MAX_NTHREADS];
    DWORD thread[MAX_NTHREADS];

    for (j = 0; j < nthreads; j++)
    {
        hthread[j] = CreateThread(NULL, 0, fns[j], args[j], 0, thread + j);
        if (hthread[j] == NULL)
        {
            printf("CreateThread fails\n");
            exit(1);
        }
    }

    WaitForMultipleObjects(nthreads, hthread, TRUE, INFINITE);
    for (j = 0; j < nthreads; j++)
    {
        CloseHandle(hthread[j]);
    }
    return ret;
#else
    for (j = 0; j < nthreads; j++)
        (fns[j]) (args[j]);
#endif
    return ret;
}
