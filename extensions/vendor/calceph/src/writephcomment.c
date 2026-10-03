/*-----------------------------------------------------------------*/
/*!
  \file writephspk2.c
  \brief perform the writing of a comment to a spk file.

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

#define __CALCEPH_WITHIN_CALCEPH 1

#include "writephinternal.h"

/*--------------------------------------------------------------------------*/
/*!
    write a comment to a spk file, the comment must be written just after
    the header of the file, otherwise an error is returned.

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param comment (in) the comment to write
*/
/*--------------------------------------------------------------------------*/
int writeph_comment(t_writephbin *eph, const char *comment)
{
    buffer_error_t buffer_error;

    if (eph == NULL || comment == NULL)
    {
        fatalerror("writeph_comment : NULL pointer eph or comment\n");
        return 0;
    }

    if (writeph_word2rec(eph->header.free) != 2)
    {
        fatalerror("writeph_comment : can't write the comment at record %d of "
                   "the ephemeris file '%s' (the comment must be written just after the header)\n",
                   writeph_word2rec(eph->header.free), eph->header.ifname);
        return 0;
    }

    size_t len = strlen(comment);
    int nrec;

    nrec = (len - 1) / DAF_RECORD_LEN + 1;  /* number of records to write */

    char *padded_comment = (char *) malloc(nrec * DAF_RECORD_LEN);

    if (padded_comment == NULL)
    {
        fatalerror("writeph_comment : memory allocation error\n System error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        return 0;
    }
    memset(padded_comment, ' ', nrec * DAF_RECORD_LEN);
    memcpy(padded_comment, comment, len);

    if (!writeph_pwrite(eph, padded_comment, nrec * DAF_RECORD_LEN, (off_t) DAF_RECORD_LEN, 0))
    {
        free(padded_comment);
        return 0;
    }
    free(padded_comment);

    /* update the first free word address */
    eph->header.free += nrec * WORD_PER_RECORD;

    return 1;
}
