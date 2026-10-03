/*-----------------------------------------------------------------*/
/*!
  \file writephcreateopenclose.c
  \brief perform the basic I/O operations of the writing 
         module to binary (DAF) ephemeris files.
         (i.e. opening, creating, and closing the file.)

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
/*! open a binary file for writing

    return NULL on error.
    return a descriptor of the ephemeris on success.

  @param filename (in) name/path of the file to open
*/
/*--------------------------------------------------------------------------*/
static t_writephbin *writeph_open(const char *filename)
{
    buffer_error_t buffer_error;

    if (!filename)
    {
        fatalerror("writeph_open: NULL pointer as input argument\n");
        return NULL;
    }

    /* allocate memory for the descriptor */
    t_writephbin *res = calloc(1, sizeof(t_writephbin));

    if (res == NULL)
    {
        fatalerror("writeph_open: can't allocate memory for t_writephbin\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));

        return NULL;
    }

    /* open the file in binary read/write mode */
    FILE *file = fopen(filename, "rb+");

    if (!file)
    {
        fatalerror("writeph_open: unable to open ephemeris file: '%s'\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        free(res);
        return NULL;
    }

    /* read the header and store it in the descriptor */
    struct SPKHeader header;

    if (fread(&header, sizeof(header), 1, file) != 1)
    {
        fatalerror("writeph_open: unable to read header from ephemeris file: '%s'\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        free(res);
        fclose(file);
        return NULL;
    }

    /* check the ftp string */
    if (!calceph_spk_ftp(header.ftp))
    {
        fatalerror("writeph_open: the FTP string is not valid in the file '%s'\n", filename);
        free(res);
        fclose(file);
        return NULL;
    }
    res->file = file;
    res->header = header;
    res->reservations = NULL;
    res->gen_segs = NULL;
    calceph_mtx_init_plain(&(res->lock));

    return res;
}

/*--------------------------------------------------------------------------*/
/*! open a binary spk file for writing

    return NULL on error.
    return a descriptor of the ephemeris on success.

  @param filename (in) name/path of the file to open
*/
/*--------------------------------------------------------------------------*/
t_writephbin *writeph_spk_open(const char *filename)
{
    buffer_error_t buffer_error;

    t_writephbin *res = writeph_open(filename);

    if (res == NULL)
        return NULL;

    /* set the fields of the descriptor */
    if (strncmp(res->header.idword, "DAF/SPK ", IDWORD_LEN) != 0)
    {
        fatalerror("writeph_spk_open: the file '%s' is not a SPK file\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        writeph_close(res);
        return NULL;
    }

    res->file_type = DAF_SPK;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! open a binary pck file for writing

    return NULL on error.
    return a descriptor of the ephemeris on success.
    
  @param filename (in) name/path of the file to open
*/
/*--------------------------------------------------------------------------*/
t_writephbin *writeph_pck_open(const char *filename)
{
    buffer_error_t buffer_error;

    t_writephbin *res = writeph_open(filename);

    if (res == NULL)
        return NULL;

    /* set the fields of the descriptor */
    if (strncmp(res->header.idword, "DAF/PCK ", IDWORD_LEN) != 0)
    {
        fatalerror("writeph_pck_open: the file '%s' is not a PCK file\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        writeph_close(res);
        return NULL;
    }

    res->file_type = DAF_PCK;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! create a binary file of any filetype for writing

    return NULL on error.
    return a descriptor of the ephemeris on success.

  @param filename (in) name/path of the file to create
  @param ifname (in) intern filename, it will be stored in the file header
  @param type (in) type of the file to create
*/
/*--------------------------------------------------------------------------*/
static t_writephbin *writeph_create(const char *filename, const char *ifname, enum SPICEfiletype type)
{
    buffer_error_t buffer_error;

    if (!filename || !ifname)
    {
        fatalerror("writeph_create: NULL pointer as input argument\n");
        return NULL;
    }

    size_t ifname_len = strlen(ifname);

    if (ifname_len > IFNAME_LEN)
    {
        fatalerror("writeph_create: ifname is too long (max length is %d)\n", IFNAME_LEN);
        return NULL;
    }

    /* allocate memory for the descriptor */
    t_writephbin *res = calloc(1, sizeof(t_writephbin));

    if (res == NULL)
    {
        fatalerror("writeph_create: can't allocate memory for t_writephbin\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));

        return NULL;
    }

    /* open the file in binary read/write mode */
    FILE *file = fopen(filename, "wb+");

    if (!file)
    {
        fatalerror("writeph_create: unable to create ephemeris file: '%s'\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        free(res);
        return NULL;
    }

    /* create a header for the descriptor */
    struct SPKHeader header;

    memset(&header, '\0', sizeof(header));
    switch (type)
    {
        case DAF_SPK:
            memcpy(header.idword, "DAF/SPK ", IDWORD_LEN);
            break;
        case DAF_PCK:
            memcpy(header.idword, "DAF/PCK ", IDWORD_LEN);
            break;
        default:
            fatalerror("writeph_create: unsupported file type %d\n", type);
            free(res);
            fclose(file);
            return NULL;
    }
    memcpy(header.ifname, ifname, ifname_len);
    memcpy(header.bff, writeph_getbff(), BFF_LEN);

    const unsigned char validftp[FTP_LEN] = {
        'F', 'T', 'P', 'S', 'T', 'R', ':', 13, ':', 10, ':', 13, 10, ':',
        13, 0, ':', 129, ':', 16, 206, ':', 'E', 'N', 'D', 'F', 'T', 'P'
    };

    memcpy(header.ftp, validftp, FTP_LEN);

    header.nd = 2;
    header.ni = 6;
    header.fwd = 0;
    header.bwd = 0;
    header.free = writeph_rec2word(2);

    if (fwrite(&header, sizeof(header), 1, file) != 1)
    {
        fatalerror("writeph_create: unable to write a header into ephemeris file: '%s'\nSystem error : '%s'\n",
                   filename, calceph_strerror_errno(buffer_error));
        free(res);
        fclose(file);
        return NULL;
    }

    res->header = header;
    res->file_type = type;
    res->file = file;
    res->reservations = NULL;
    res->gen_segs = NULL;
    calceph_mtx_init_plain(&(res->lock));

    return res;
}

/*--------------------------------------------------------------------------*/
/*! create a binary spk file for writing

    return NULL on error.
    return a descriptor of the ephemeris on success.

  @param filename (in) name/path of the file to create
  @param ifname (in) intern filename, it will be stored in the file header
  @param flags (in) reserved for future extension (.e.g, 64-bit file support, locking, ...)
  should be 0

*/
/*--------------------------------------------------------------------------*/
t_writephbin *writeph_spk_create(const char *filename, const char *ifname, int flags)
{
    (void)flags;
    return writeph_create(filename, ifname, DAF_SPK);
}

/*--------------------------------------------------------------------------*/
/*! create a binary pck file for writing

   return NULL on error.
   return a descriptor of the ephemeris on success.

  @param filename (in) name/path of the file to create
  @param ifname (in) intern filename, it will be stored in the file header
  @param flags (in) reserved for future extension (.e.g, 64-bit file support, locking, ...)
  should be 0
*/
/*--------------------------------------------------------------------------*/
t_writephbin *writeph_pck_create(const char *filename, const char *ifname, int flags)
{
    (void)flags;
    return writeph_create(filename, ifname, DAF_PCK);
}

/*--------------------------------------------------------------------------*/
/*! close a binary (DAF) ephemeris file created/opened for writing

    return 0 on error.
    return 1 on success.

  @param eph (in) descriptor of the ephemeris file to close
*/
/*--------------------------------------------------------------------------*/
int writeph_close(t_writephbin *eph)
{
    buffer_error_t buffer_error;

/* GCOVR_EXCL_START */
    if (!eph)
    {
        fatalerror("writeph_close: NULL pointer as input argument\n");
        return 0;
    }
/* GCOVR_EXCL_STOP */

    /* close the file */
    if (fclose(eph->file) == EOF)
    {
        fatalerror("writeph_close: unable to close ephemeris file: '%s'\nSystem error : '%s'\n", eph->header.ifname,
                   calceph_strerror_errno(buffer_error));
        return 0;
    }

    /* free the reservations and trajectories */
    writeph_free_reservations(eph->reservations);

    /* free the partitions */
    if (eph->gen_segs)
    {
        free(eph->gen_segs->epochs);
        free(eph->gen_segs);
    }

    /* destroy the mutex */
    calceph_mtx_destroy(&(eph->lock));

    /* free the descriptor */
    free(eph);

    return 1;
}
