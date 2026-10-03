
/*-----------------------------------------------------------------*/
/*!
  \file writephcreateopencommentfail.c
  \brief check the fail cases of the open, create, and comment functionalities of writeph.

  \author  A. Durst, M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2025-2026, CNRS
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

#include "calcephconfig.h"
#include "calceph.h"

#include "openfiles.h"

static void hidemsg(const char *msg);

/*-----------------------------------------------------------------*/
/* function to hide the error message */
/*-----------------------------------------------------------------*/
static void hidemsg(const char *PARAMETER_UNUSED(msg))
{
#if HAVE_PRAGMA_UNUSED
#pragma unused(msg)
#endif
    /* printf("msg='%s'\n", msg); */
}

int main(void)
{
    /* hide the error messages */   
    calceph_seterrorhandler(3, hidemsg);

    if (writeph_spk_open("this_file_does_not_exist.bsp") != NULL)
    {
        printf("Error: opening a non-existing file does not fail\n");
        return 1;
    }
    if (writeph_spk_open(NULL) != NULL)
    {
        printf("Error: opening a NULL file does not fail\n");
        return 1;
    }
    if (writeph_spk_create(NULL, "name",0) != NULL)
    {
        printf("Error: creating a NULL file does not fail\n");
        return 1;
    }
    if (writeph_spk_create("file.bsp", NULL,0) != NULL)
    {
        printf("Error: creating a file with a NULL internal name does not fail\n");
        return 1;
    }

    t_writephbin *w_eph = writeph_spk_create("writephcreateopencommentfail.bsp", "name", 0);
    if (!w_eph)
    {
        printf("Error: creating the file failed\n");
        return 1;
    }
    if (writeph_comment(NULL, "This is a comment"))
    {
        printf("Error: writing a comment with a NULL descriptor does not fail\n");
        return 1;
    }
    if (writeph_comment(w_eph, NULL) != 0)
    {
        printf("Error: writing a NULL comment does not fail\n");
        return 1;
    }

    /* try to write a comment after writing a segment */
    const double coefs[] = {1, 2, 3};
    if (!writeph_spk2_seq_write(w_eph, 299, 5, 1, 2451545, 0, 2451929, 0, 32, coefs, 1, 0, "segid"))
    {
        printf("Error: writing the segment failed\n");
        return 1;
    }
    if (writeph_comment(w_eph, "This is a comment"))
    {
        printf("Error: writing a comment after a segment does not fail\n");
        return 1;
    }

    if (!writeph_close(w_eph))
    {
        printf("Error: closing the file failed\n");
        return 1;
    }

    return 0;
}