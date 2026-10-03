/*-----------------------------------------------------------------*/
/*!
  \file writephdumprestore.c
  \brief perform the saving and the recovering of a binary (DAF)
         ephemeris file when its writing is interrupted.

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
/*! save the state of a binary (DAF) ephemeris file to a dump file

    return 0 on error.
    return 1 on success.

  @param eph (in) descriptor of the ephemeris file to close
  @param filename (in) path of the dump file to store the epheremis state
*/
/*--------------------------------------------------------------------------*/
int writeph_dump(t_writephbin *eph, const char *filename)
{
    buffer_error_t buffer_error;

    if (!eph || !filename)
    {
        fatalerror("writeph_dump: NULL pointer as input argument\n");
        return 0;
    }

    FILE *dump_file = fopen(filename, "wb");

    if (!dump_file)
    {
        fatalerror("writeph_dump: unable to create dump file: '%s'\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        return 0;
    }

    /* leave space for the dump header */
    if (fseeko(dump_file, sizeof(struct dump_header), SEEK_SET) != 0)
    {
        fatalerror("writeph_dump: unable to seek to the beginning of the dump file\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        fclose(dump_file);
        return 0;
    }

    /* flush the ephemeris file */
    fflush(eph->file);

    /* write all the reservations */
    struct reservation *reserv = eph->reservations;
    int count = 0;

    while (reserv)
    {
        count++;
        /* write the reservation data without the .trajectories and .next field */
        if (fwrite(reserv, sizeof(struct reservation) - 2 * sizeof(void *), 1, dump_file) != 1)
        {
            fatalerror("writeph_dump: unable to write reservation data\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            fclose(dump_file);
            return 0;
        }
        if (fwrite(reserv->trajectories, sizeof(struct trajectory), reserv->target_count, dump_file) !=
            (size_t) reserv->target_count)
        {
            fatalerror("writeph_dump: unable to write trajectory data\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            fclose(dump_file);
            return 0;
        }
        reserv = reserv->next;
    }

    /* prepare the dump header */
    struct dump_header dmp_header;

    dmp_header.location[0] = eph->header.bwd;
    dmp_header.location[1] = eph->header.free;
    if (eph->header.bwd != 0)
    {
        /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
        off_t offset = ((off_t) eph->header.bwd - 1) * DAF_RECORD_LEN;

        if (!writeph_pread(eph, dmp_header.descriptor, 3 * sizeof(double), offset, 0))
        {
            fclose(dump_file);
            return 0;
        }
    }
    dmp_header.n_reservations = count;

    /* go back to the beginning of the dump file */
    if (fseeko(dump_file, 0, SEEK_SET) != 0)
    {
        fatalerror("writeph_dump: unable to seek to the beginning of the dump file\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        fclose(dump_file);
        return 0;
    }

    /* write the dump header */
    if (fwrite(&dmp_header, sizeof(struct dump_header), 1, dump_file) != 1)
    {
        fatalerror("writeph_dump: unable to write dump header\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        fclose(dump_file);
        return 0;
    }

    fclose(dump_file);
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! restore the state of a binary (DAF) ephemeris file from a dump file

    return 0 on error.
    return 1 on success.

  @param eph (in) descriptor of the ephemeris file to close
  @param filename (in) path of the dump file
*/
/*--------------------------------------------------------------------------*/
int writeph_restore(t_writephbin *eph, const char *filename)
{
    buffer_error_t buffer_error;

    if (!eph || !filename)
    {
        fatalerror("writeph_restore: NULL pointer as input argument\n");
        return 0;
    }

    FILE *dump_file = fopen(filename, "rb");

    if (!dump_file)
    {
        fatalerror("writeph_restore: unable to open dump file: '%s'\nSystem error : '%s'\n", filename,
                   calceph_strerror_errno(buffer_error));
        return 0;
    }

    /* read the dump header */
    struct dump_header dmp_header;

    if (fread(&dmp_header, sizeof(struct dump_header), 1, dump_file) != 1)
    {
        fatalerror("writeph_restore: unable to read dump header\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        fclose(dump_file);
        return 0;
    }

    /* read all the reservations from the dump file */
    struct reservation *reservations = NULL;
    struct reservation *last_reserv = NULL;
    int j;

    for (j = 0; j < dmp_header.n_reservations; j++)
    {
        struct reservation *reserv = calloc(1, sizeof(struct reservation));

        if (!reserv)
        {
            fatalerror("writeph_restore: unable to allocate memory\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            writeph_free_reservations(reservations);
            fclose(dump_file);
            return 0;
        }

        /* link the new reservation into the list */
        if (last_reserv)
            last_reserv->next = reserv;
        else
            reservations = reserv;

        /* read the reservation data without the .trajectories and .next field */
        if (fread(reserv, sizeof(struct reservation) - 2 * sizeof(void *), 1, dump_file) != 1)
        {
            fatalerror("writeph_restore: unable to read reservation data\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            writeph_free_reservations(reservations);
            fclose(dump_file);
            return 0;
        }

        /* allocate memory for the trajectories */
        reserv->trajectories = malloc(reserv->target_count * sizeof(struct trajectory));
        if (!reserv->trajectories)
        {
            fatalerror("writeph_restore: unable to allocate memory\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            writeph_free_reservations(reservations);
            fclose(dump_file);
            return 0;
        }
        /* read the trajectories data */
        if (fread(reserv->trajectories, sizeof(struct trajectory), reserv->target_count, dump_file) !=
            (size_t) reserv->target_count)
        {
            fatalerror("writeph_restore: unable to read trajectory data\nSystem error : '%s'\n",
                       calceph_strerror_errno(buffer_error));
            writeph_free_reservations(reservations);
            fclose(dump_file);
            return 0;
        }

        reserv->next = NULL;
        last_reserv = reserv;
    }
    fclose(dump_file);

    /* restore the header information */
    eph->header.bwd = dmp_header.location[0];
    eph->header.free = dmp_header.location[1];
    if (!writeph_header(eph, 0))
    {
        writeph_free_reservations(reservations);
        return 0;
    }

    /* restore the descriptor of the last summary record */
    if (eph->header.bwd != 0)
    {
        /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
        off_t offset = ((off_t) eph->header.bwd - 1) * DAF_RECORD_LEN;

        if (!writeph_pwrite(eph, dmp_header.descriptor, 3 * sizeof(double), offset, 0))
        {
            writeph_free_reservations(reservations);
            return 0;
        }
    }

    /* restore the reservations */
    eph->reservations = reservations;

    return 1;
}
