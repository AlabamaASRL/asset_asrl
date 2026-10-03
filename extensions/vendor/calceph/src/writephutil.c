/*-----------------------------------------------------------------*/
/*!
  \file writephutil.c
  \brief additionnal private tools for the writeph module.

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

#include "writephinternal.h"

/*--------------------------------------------------------------------------*/
/*!
    write to a file at a given offset, using pwrite if available
    or fseeko/fwrite if not.

    return 0 on error.
    return 1 on success.

  @param file (in) file descriptor
  @param buf (in) buffer to write
  @param count (in) number of bytes to write
  @param offset (in) offset in the file where to write
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_pwrite(t_writephbin *eph, const void *buf, size_t count, off_t offset, int para)
{
    buffer_error_t buffer_error;
    int ret = 1, fret;

    if (para)
    {
#if HAVE_UNISTD_H && HAVE_PWRITE
        if (pwrite(fileno(eph->file), buf, count, offset) != (ssize_t) count)
        {
            fatalerror("writeph_pwrite: can't write %zu bytes at offset %lld\nSystem error : '%s'\n", count,
                       (long long) offset, calceph_strerror_errno(buffer_error));
            return 0;
        }
        return 1;
#elif !CALCEPH_HAVE_MUTEX
        fatalerror("writeph_pwrite: neither pwrite nor pthread are available on this system\n");
        return 0;
#endif
    }

    if (para)
    {
        calceph_mtx_lock(&(eph->lock));
    }

    fret = fseeko(eph->file, offset, SEEK_SET);
    if (fret != 0)
    {
        fatalerror("writeph_pwrite: can't seek to offset %lld\nSystem error : '%s'\n", (long long) offset,
                   calceph_strerror_errno(buffer_error));
        ret = 0;
    }
    if (ret == 1)
    {
        if (fwrite(buf, 1, count, eph->file) != count)
        {
            fatalerror("writeph_pwrite: can't write %zu bytes at offset %lld\nSystem error : '%s'\n", count,
                       (long long) offset, calceph_strerror_errno(buffer_error));
            ret = 0;
        }
    }

    if (para)
    {
        calceph_mtx_unlock(&(eph->lock));
    }
    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
    read from a file at a given offset, using pread if available
    or fseeko/fread if not.

    return 0 on error.
    return 1 on success.

  @param file (in) file descriptor
  @param buf (out) buffer to read into
  @param count (in) number of bytes to read
  @param offset (in) offset in the file where to read
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_pread(t_writephbin *eph, void *buf, size_t count, off_t offset, int para)
{
    int ret = 1, fret;
    buffer_error_t buffer_error;

    if (para)
    {
#if HAVE_UNISTD_H && HAVE_PREAD
        if (pread(fileno(eph->file), buf, count, offset) != (ssize_t) count)
        {
            fatalerror("writeph_pread: can't read %zu bytes at offset %lld\nSystem error : '%s'\n", count,
                       (long long) offset, calceph_strerror_errno(buffer_error));
            return 0;
        }
        return 1;
#elif !CALCEPH_HAVE_MUTEX
        fatalerror("writeph_pread: neither pread nor pthread are available on this system\n");
        return 0;
#endif
    }

    if (para)
    {
        calceph_mtx_lock(&(eph->lock));
    }

    fret = fseeko(eph->file, offset, SEEK_SET);
    if (fret != 0)
    {
        fatalerror("writeph_pread: can't seek to offset %lld\nSystem error : '%s'\n", (long long) offset,
                   calceph_strerror_errno(buffer_error));
        ret = 0;
    }
    if (ret == 1)
    {
        if (fread(buf, 1, count, eph->file) != count)
        {
            fatalerror("writeph_pread: can't write %zu bytes at offset %lld\nSystem error : '%s'\n", count,
                       (long long) offset, calceph_strerror_errno(buffer_error));
            ret = 0;
        }
    }

    if (para)
    {
        calceph_mtx_unlock(&(eph->lock));
    }

    return 1;
}

/*--------------------------------------------------------------------------*/
/*
     return the bfs of the OS
*/
/*--------------------------------------------------------------------------*/
const char *writeph_getbff(void)
{
    unsigned int x = 0x01000002;
    unsigned char *c = (unsigned char *) &x;

    if (*c == 0x01)
    {
        return "BIG-IEEE";
    }
    else if (*c == 0x02)
    {
        return "LTL-IEEE";
    }
    else
    {
        return "UNKWOWN_";
    }
}

/*--------------------------------------------------------------------------*/
/*!
    return the first word address of the corresponding record
    @param rec_addr (in) the record address (number)
*/
/*--------------------------------------------------------------------------*/
int writeph_rec2word(int rec_addr)
{
    /* check for an overflow */
    if (rec_addr > INT_MAX / WORD_PER_RECORD)
    {
        fatalerror("writeph_rec2word: record address %d is too large\n", rec_addr);
        return 0;
    }
    return (rec_addr - 1) * WORD_PER_RECORD + 1;
}

/*--------------------------------------------------------------------------*/
/*!
    return the record number corresponding to the word address
    @param word_addr (in) the word address
*/
/*--------------------------------------------------------------------------*/
int writeph_word2rec(int word_addr)
{
    return (word_addr - 1) / WORD_PER_RECORD + 1;
}

/*--------------------------------------------------------------------------*/
/*!
    return the current record address of the file
    @param eph (in) ephemermis descriptor
*/
/*--------------------------------------------------------------------------*/
int writeph_curr(t_writephbin *eph)
{
    return writeph_word2rec(eph->header.free);
}

/*--------------------------------------------------------------------------*/
/*!
    pad the current record with zeros up to the end of the record

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_padrec(t_writephbin *eph, char pad, int para)
{
    buffer_error_t buffer_error;

    if ((eph->header.free - 1) % WORD_PER_RECORD != 0)
    {
        int nword_to_pad;

        nword_to_pad = WORD_PER_RECORD - (eph->header.free - 1) % WORD_PER_RECORD;
        double *padbuf = (double *) malloc(nword_to_pad * sizeof(double));

        if (!padbuf)
        {
            fatalerror("writeph_padrec: memory allocation error\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            return 0;
        }
        memset(padbuf, pad, nword_to_pad * sizeof(double));

        /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
        off_t offset = (off_t) (eph->header.free - 1) * (off_t) sizeof(double);

        if (!writeph_pwrite(eph, padbuf, nword_to_pad * sizeof(double), offset, para))
        {
            free(padbuf);
            return 0;
        }
        eph->header.free += nword_to_pad;
        free(padbuf);
    }
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! write to the header of the file to either create or update it

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_header(t_writephbin *eph, int para)
{
    return writeph_pwrite(eph, &(eph->header), sizeof(eph->header), 0, para);
}

/*--------------------------------------------------------------------------*/
/*!
    write to the first free record of the file

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param ptr (in) location of the data to write
  @param size (in) size of each element to write
  @param nmemb (in) number of elements to write
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_rec(t_writephbin *eph, const void *ptr, size_t size, size_t nmemb, char pad, int para)
{
#if DEBUG
    if (size == 0 || nmemb == 0)
    {
        fatalerror("writeph_rec: invalid size or nmemb\n");
        return 0;
    }
    if ((size * nmemb) % sizeof(double) != 0)
    {
        fatalerror("writeph_rec: size*nmemb must be a multiple of sizeof(double)\n");
        return 0;
    }
#endif

    /* only one record can be written at a time */
    if (size > DAF_RECORD_LEN)
    {
        fatalerror("Can't write at record %d of "
                   "the ephemeris file '%s' (the data cannot fit in one record)\n",
                   writeph_word2rec(eph->header.free), eph->header.ifname);
        return 0;
    }

    /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
    off_t offset = (off_t) (eph->header.free - 1) * (off_t) sizeof(double);

    /* write the data */
    if (!writeph_pwrite(eph, ptr, size * nmemb, offset, para))
        return 0;

    /* update the first free word */
    eph->header.free += (int) ((size * nmemb) / sizeof(double));

    /* padding */
    if (!writeph_padrec(eph, pad, para))
        return 0;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*!
       write a segment id to the first free record of the file
       
       return 0 on error.
       return 1 on success.
       
         @param eph (in) ephemermis descriptor
         @param segid (in) segment id string
         @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_segid(t_writephbin *eph, const char *segid, int para)
{
    buffer_error_t buffer_error;

    size_t segid_len = strlen(segid);

    /* check the segment id length */
    if (segid_len > SEGMENTID_LEN)
    {
        fatalerror("writeph_segid: the segment id has to be less than %d characters\n", SEGMENTID_LEN);
        return 0;
    }

    /* allocate the padded segment id */
    char *padded_segid = malloc(SEGMENTID_LEN * sizeof(char));

    if (!padded_segid)
    {
        fatalerror("writeph_segid: error while allocating the padded segment id\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        return 0;
    }

    /* copy the segment id into it and pad the rest with spaces */
    size_t j;

    for (j = 0; j < SEGMENTID_LEN; j++)
        padded_segid[j] = (j < segid_len) ? segid[j] : ' ';

    /* add the segment id to the first free record */
    if (writeph_rec(eph, padded_segid, SEGMENTID_LEN * sizeof(char), 1, ' ', para) != 1)
    {
        free(padded_segid);
        return 0;
    }

    free(padded_segid);

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! 
    add a summary record to the list of summary records, creating the list 
    if needed

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------*/
int writeph_list(t_writephbin *eph, int para)
{
    /* write the summary record */
    double summary_record_header[3] = { 0, eph->header.bwd, 1 };
    if (writeph_rec(eph, summary_record_header, sizeof(double), 3, '\0', para) != 1)
        return 0;

    /* insert it into the list */
    if (eph->header.bwd == 0)
    {
        eph->header.bwd = writeph_word2rec(eph->header.free) - 1;
        eph->header.fwd = eph->header.bwd;
    }
    else
    {
        /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
        off_t offset = (off_t) (eph->header.bwd - 1) * DAF_RECORD_LEN;

        if (!writeph_pread(eph, summary_record_header, 3 * sizeof(double), offset, para))
            return 0;
        summary_record_header[0] = writeph_word2rec(eph->header.free) - 1;

        if (!writeph_pwrite(eph, summary_record_header, 3 * sizeof(double), offset, para))
            return 0;
        eph->header.bwd = summary_record_header[0];
    }

    return 1;
}

/*--------------------------------------------------------------------------- */
/*! 
    write a segment descriptor to a file

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemeris descriptor
  @param seg_des (in) struct containing the segment descriptor data
  @param size (in) size in words of the segment to be written
  @param para (in) if different from 0, use parallel safe I/O functions
*/
/*--------------------------------------------------------------------------- */
int writeph_segdescriptor(t_writephbin *eph, struct segment_descriptor seg_des, int size, int para)
{
#if DEBUG
    if (size < 0)
    {
        fatalerror("writeph_segdescriptor: invalid segment size\n");
        return 0;
    }
#endif
    /* number of segment descriptor in the last summary record */
    double count = 0;

    if (eph->header.bwd == 0)   /* If there is no summary record list yet then create it */
    {
        if (!writeph_list(eph, 0))
            return 0;
    }
    else                        /* If the summary record list already exist */
    {
        /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
        off_t offset = (off_t) (eph->header.bwd - 1) * DAF_RECORD_LEN;

        /* read the last summary record information */
        double summary_record_header[3];

        if (!writeph_pread(eph, summary_record_header, 3 * sizeof(double), offset, 0))
            return 0;
        count = summary_record_header[2];

        /* if the last summary record is full add a new summary record to the list */
        if (count >= NSEGMENT_PERRECORD)
        {
            if (!writeph_list(eph, 0))
                return 0;
            count = 0;
        }
    }

    /* set the segment bounds */
    if (seg_des.i6 != -1)
    {
        /* for spk segments */
        seg_des.i5 = eph->header.free + WORD_PER_RECORD;
        seg_des.i6 = seg_des.i5 + size - 1;
    }
    else
    {
        /* for pck segments */
        seg_des.i4 = eph->header.free + WORD_PER_RECORD;
        seg_des.i5 = seg_des.i4 + size - 1;
    }

    /* compute the offsets, make sure they are of type off_t to prevent an int overflow */
    off_t offset1 =
        (off_t) (eph->header.bwd - 1) * DAF_RECORD_LEN + 3 * (off_t) sizeof(double) +
        count * (off_t) sizeof(struct segment_descriptor);
    off_t offset2 = (off_t) (eph->header.bwd - 1) * DAF_RECORD_LEN + 2 * (off_t) sizeof(double);

    if (eph->gen_segs)
    {
        eph->gen_segs->seg_des_bound_fields_address = (int) (offset1 / sizeof(double)) + 4;
    }

    /* write the segment descriptor */
    if (writeph_pwrite(eph, &seg_des, sizeof(seg_des), offset1, para) != 1)
        return 0;

    /* update the segment count */
    count++;
    if (!writeph_pwrite(eph, &count, sizeof(double), offset2, para))
        return 0;

    return 1;
}

/*--------------------------------------------------------------------------- */
/*!
    free all the reservations

  @param reserv (in) pointer to the first reservation to free
*/
/*--------------------------------------------------------------------------- */
void writeph_free_reservations(struct reservation *reserv)
{
    while (reserv)
    {
        struct reservation *next = reserv->next;

        free(reserv->trajectories);
        free(reserv);
        reserv = next;
    }
}
