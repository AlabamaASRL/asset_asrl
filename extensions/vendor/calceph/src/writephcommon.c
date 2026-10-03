/*-----------------------------------------------------------------*/
/*!
  \file writephcommon.c
  \brief Common functions for writing ephemeris files in binary format.

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

/*---------------------------------------------------------------------------*/
/*!
    Convert TCB time in seconds to TDB time in seconds.

    The constants come from the IAU 2006 Resolution B3
    Resolution B3: "Re-definition of Barycentric Dynamical Time, TDB"

    The implementation is done using the equations 22 (same as IAU) of
    @article{
        Turyshev_2025,
        doi = {10.3847/1538-4357/adcc18},
        url = {https://dx.doi.org/10.3847/1538-4357/adcc18},
        year = {2025},
        month = {may},
        publisher = {The American Astronomical Society},
        volume = {985},
        number = {1},
        pages = {140},
        author = {Turyshev, Slava G. and Williams, James G. and Boggs, Dale H. and Park,
        Ryan S.},
        title = {Relativistic Time Transformations between the Solar System Barycenter, 
        Earth, and Moon},
        journal = {The Astrophysical Journal},
    }

    or same (page 17) in
    @ARTICLE{
        2010ITN....36....1P,
        author = {{Petit}, G{\'e}rard and {Luzum}, Brian},
            title = "{IERS Conventions (2010)}",
        journal = {IERS Technical Note},
            year = 2010,
            month = jan,
        volume = {36},
            pages = {1},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2010ITN....36....1P},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    cf  IAU 2006 resolution B3 : TDB = TCB - LB*(TCB-T0)*86400 + TDB0  
                            => TCB = TDB + (  LB*(TDB-T0)*86400 - TDB0 ) / (
    1-LB) 
*/
static double writeph_tcb2tdb(double tcb_sec)
{
    double Lb = 1.550519768E-8;
    double T0 = 2443144.5003725;
    double delta0 = -6.55E-5;
    double tcb_julian = tcb_sec / 86400.0 + 2451545.0;

    return tcb_sec - Lb * (tcb_julian - T0) * 86400.0 + delta0;
}

/* calculate the number of directory records */
static int writeph_ndir(int record_count)
{
    int n_dir = record_count / 100;

    if (record_count % 100 == 0)
        n_dir--;
    return n_dir;
}

/* return 1 if the file can fit the data, 0 otherwise */
static int writeph_checksize(enum SPKdatatype data_type, int free, int target_count, const int *record_counts, int deg)
{
    /* compute the size needed */
    size_t total_size = 0;
    size_t i;

    for (i = 0; i < (size_t) target_count; i++)
    {
        size_t segment_words;
        size_t segment_recs;
        size_t record_count = record_counts[i];

        switch (data_type)
        {
            case SPK_SEGTYPE2:
            case SPK_SEGTYPE102:
                segment_words = record_count * (2 + 3 * (deg + 1)) + 4;
                break;
            case SPK_SEGTYPE3:
            case SPK_SEGTYPE103:
                segment_words = record_count * (2 + 6 * (deg + 1)) + 4;
                break;
            case SPK_SEGTYPE8:
            case SPK_SEGTYPE12:
                segment_words = record_count * 6 + 4;
                break;
            case SPK_SEGTYPE9:
            case SPK_SEGTYPE13:
                segment_words = record_count * 6 + record_count + writeph_ndir((int) record_count) + 2;
                break;
            case SPK_SEGTYPE14:
                segment_words = record_count * (6 * (deg + 1));
                break;
            default:
                fatalerror("writeph_checksize: unsupported segment type %d\n", data_type);
                return 0;
        }
        segment_recs = (segment_words - 1) / WORD_PER_RECORD + 1;

        total_size += (segment_recs + 1) * WORD_PER_RECORD; /* add segment payload and id */
    }
    total_size += (((size_t) target_count - 1) / NSEGMENT_PERRECORD + 1) * WORD_PER_RECORD; /* add summary records */

    if ((size_t) free + total_size > (size_t) (INT_MAX))
        return 0;
    return 1;
}

/* initialize a segment, by writing the segment descriptor and id,
   returns the size of the segment in words on success, 0 on error */
static int writeph_init
    (enum SPICEfiletype file_type,
     double start_sec,
     double end_sec,
     int target,
     int center,
     int frame, enum SPKdatatype data_type, int record_count, int deg, const char *segid, t_writephbin * eph)
{
    /* prepare the segment descriptor */
    struct segment_descriptor seg_des;

    switch (file_type)
    {
        case DAF_SPK:
            seg_des = (struct segment_descriptor)
            {
                start_sec,
                end_sec,
                target,
                center,
                frame,
                (int) data_type,
                0,              /* first word, will be set later */
                0               /* last word, will be set later */
            };
            break;
        case DAF_PCK:
            seg_des = (struct segment_descriptor)
            {
                start_sec,
                end_sec,
                target,
                frame,
                (int) data_type,
                0,              /* first word, will be set later */
                0,              /* last word, will be set later */
                -1              /* unused */
            };
            break;
        default:
            fatalerror("writeph_init: unsupported file type %d\n", file_type);
            return 0;
    }

    /* size in words of the segment to be written */
    int segment_size;

    switch (data_type)
    {
        case SPK_SEGTYPE2:
        case SPK_SEGTYPE102:
            segment_size = record_count * (2 + 3 * (deg + 1)) + 4;
            break;
        case SPK_SEGTYPE3:
        case SPK_SEGTYPE103:
            segment_size = record_count * (2 + 6 * (deg + 1)) + 4;
            break;
        case SPK_SEGTYPE8:
        case SPK_SEGTYPE12:
            segment_size = record_count * 6 + 4;
            break;
        case SPK_SEGTYPE9:
        case SPK_SEGTYPE13:
            segment_size = record_count * 6 + record_count + writeph_ndir(record_count) + 2;
            break;
        case SPK_SEGTYPE14:
            /* the segment size can't be known for type 14 initialization */
            segment_size = 0;
            break;
        default:
            fatalerror("writeph_init: unsupported segment type %d\n", data_type);
            return 0;
    }

    /* add the segment descriptor to the summary record */
    if (!writeph_segdescriptor(eph, seg_des, segment_size, 0))
        return 0;

    /* add the segment id to the first free record */
    if (!writeph_segid(eph, segid, 0))
        return 0;

    return (data_type == SPK_SEGTYPE14 ? 1 : segment_size);
}

/*---------------------------------------------------------------------------*/
/*!
    write a segment of any type to a spk file in sequential mode.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
    @param data_type (in) segment type
    @param target (in) target id
    @param center (in) center id
    @param frame (in) reference frame id
    @param start_jd0 (in) start time of the segment in days (JD0 part)
    @param start_frac (in) start time of the segment in days (fractional part)
    @param end_jd0 (in) end time of the segment in days (JD0 part)
    @param end_frac (in) end time of the segment in days (fractional part)
    @param intlen_jd (in) interpolation interval length,
                              in days (for fixed time step, unused otherwise)
    @param data (in) array of data to write
    @param epochs (in) array of epochs in days (for unequal time steps, unused otherwise)
    @param record_count (in) number of records in the segment
    @param deg (in) degree of the polynomials
    @param segid (in) segment identifier
*/
/*---------------------------------------------------------------------------*/
int writeph_seq_write
    (t_writephbin * eph,
     enum SPICEfiletype file_type,
     enum SPKdatatype data_type,
     int target,
     int center,
     int frame,
     double start_jd0,
     double start_frac,
     double end_jd0,
     double end_frac,
     double intlen_jd, const double *data, const double *epochs, int record_count, int deg, const char *segid)
{
    buffer_error_t buffer_error;

    /* check the ephemeris descriptor */
    if (!eph)
    {
        fatalerror("writeph_seq_write: wrong parameter, eph is NULL\n");
        return 0;
    }

    /* check if a generic segment is being written */
    if (eph->gen_segs != NULL)
    {
        fatalerror
            ("writeph_seq_write: a generic segment is being written, you must end it before writing another segment\n");
        return 0;
    }

    /* check if there is enough space in the ephemeris file */
    if (!writeph_checksize(data_type, eph->header.free, 1, &record_count, deg))
    {
        fatalerror("writeph_seq_write: not enough space in the ephemeris file '%s' to write the segment\n",
                   eph->header.ifname);
        return 0;
    }

    /* convert times to seconds from J2000 */
    int jd2000 = 2451545;
    double start_sec = ((start_jd0 - jd2000) + start_frac) * 86400;
    double end_sec = ((end_jd0 - jd2000) + end_frac) * 86400;
    double intlen_sec = intlen_jd * 86400;

    /* convert TCB to TDB if needed */
    double start_sec_tdb = start_sec;
    double end_sec_tdb = end_sec;

    if (data_type == SPK_SEGTYPE102 || data_type == SPK_SEGTYPE103 || data_type == SPK_SEGTYPE120)
    {
        start_sec_tdb = writeph_tcb2tdb(start_sec);
        end_sec_tdb = writeph_tcb2tdb(end_sec);
    }

    /* initialize the segment */
    if (!writeph_init
        (file_type, start_sec_tdb, end_sec_tdb, target, center, frame, data_type, record_count, deg, segid, eph))
        return 0;

    int rsize = 0;

    /* write all the records of the segment */
    switch (data_type)
    {
        case SPK_SEGTYPE2:
        case SPK_SEGTYPE3:
        case SPK_SEGTYPE102:
        case SPK_SEGTYPE103:
            {
                double radius = intlen_sec / 2;
                int set_size;

                set_size = (data_type == SPK_SEGTYPE2 || data_type == SPK_SEGTYPE102 ? 3 : 6) * (deg + 1);
                rsize = set_size + 2;

                double *record = malloc(rsize * sizeof(double));
                int j;

                /* write one record after another */
                for (j = 0; j < record_count; j++)
                {
                    /* prepare the record */
                    double mid;

                    mid = start_sec + (j + 0.5) * intlen_sec;
                    record[0] = mid;
                    record[1] = radius;
                    memcpy(record + 2, &data[j * set_size], set_size * sizeof(double));

                    /* write the record */
                    if (fwrite(record, sizeof(double), (size_t) rsize, eph->file) != (size_t) rsize)
                    {
                        fatalerror
                            ("writephchebyshev_seq_write: can't write the data of the segment at record %d word %d of "
                             "the ephemeris file '%s'\nSystem error : '%s'\n", writeph_word2rec(eph->header.free),
                             eph->header.free, eph->header.ifname, calceph_strerror_errno(buffer_error));
                        return 0;
                    }

                    /* shift the first free word */
                    eph->header.free += rsize;
                }
                free(record);
                break;
            }
        case SPK_SEGTYPE8:
        case SPK_SEGTYPE9:
        case SPK_SEGTYPE12:
        case SPK_SEGTYPE13:
            {
                /* write all the records at once */
                if (fwrite(data, sizeof(double), record_count * 6, eph->file) != (size_t) (record_count * 6))
                {
                    fatalerror("writeph_seq_write: can't write the data of the segment at record %d word %d of "
                               "the ephemeris file '%s'\nSystem error : '%s'\n",
                               writeph_word2rec(eph->header.free), eph->header.free, eph->header.ifname,
                               calceph_strerror_errno(buffer_error));
                    return 0;
                }

                /* shift the first free word */
                eph->header.free += record_count * 6;
                break;
            }
        default:
            fatalerror("writeph_seq_write: unsupported segment type %d\n", data_type);
            return 0;
    }

    /* for unequal time step segments */
    if (data_type == SPK_SEGTYPE9 || data_type == SPK_SEGTYPE13)
    {
        /* write all the epochs */
        int j;

        for (j = 0; j < record_count; j++)
        {
            double epoch_sec = (epochs[j] - 2451545.0) * 86400.0;

            if (fwrite(&epoch_sec, sizeof(double), 1, eph->file) != 1)
            {
                fatalerror("writeph_seq_write: can't write the epoch %d of the segment at record %d word %d of "
                           "the ephemeris file '%s'\nSystem error : '%s'\n",
                           j, writeph_word2rec(eph->header.free), eph->header.free, eph->header.ifname,
                           calceph_strerror_errno(buffer_error));
                return 0;
            }
        }

        /* shift the first free word */
        eph->header.free += record_count;

        /* write the directory records */
        int n_dir = writeph_ndir(record_count);

        for (j = 1; j <= n_dir; j++)
        {
            double epoch_sec = (epochs[j * 100 - 1] - 2451545.0) * 86400.0;

            if (fwrite(&epoch_sec, sizeof(double), 1, eph->file) != 1)
            {
                fatalerror
                    ("writeph_seq_write: can't write the directory record %d of the segment at record %d word %d of "
                     "the ephemeris file '%s'\nSystem error : '%s'\n", j, writeph_word2rec(eph->header.free),
                     eph->header.free, eph->header.ifname, calceph_strerror_errno(buffer_error));
                return 0;
            }
        }

        /* shift the first free word */
        eph->header.free += n_dir;
    }

    /* write the segment metadata */
    switch (data_type)
    {
        case SPK_SEGTYPE2:
        case SPK_SEGTYPE3:
        case SPK_SEGTYPE8:
        case SPK_SEGTYPE12:
        case SPK_SEGTYPE102:
        case SPK_SEGTYPE103:
            {

                double third_value;

                if (data_type == SPK_SEGTYPE8)
                    third_value = (double) deg;
                else if (data_type == SPK_SEGTYPE12)
                    third_value = (double) ((deg + 1) / 2 - 1);
                else
                    third_value = (double) rsize;

                double directory[4] = { start_sec, intlen_sec, third_value, (double) record_count };

                if (fwrite(directory, sizeof(double), 4, eph->file) != 4)
                {
                    fatalerror("writephchebyshev_seq_write: can't write the segment directory at record %d word %d of "
                               "the ephemeris file '%s'\nSystem error : '%s'\n",
                               writeph_word2rec(eph->header.free), eph->header.free, eph->header.ifname,
                               calceph_strerror_errno(buffer_error));
                    return 0;
                }

                eph->header.free += 4;
                break;
            }
        case SPK_SEGTYPE9:
        case SPK_SEGTYPE13:
            {
                double first_value;

                if (data_type == SPK_SEGTYPE9)
                    first_value = (double) deg;
                else
                    first_value = (double) ((deg + 1) / 2 - 1);

                double directory[2] = { first_value, (double) record_count };

                if (fwrite(directory, sizeof(double), 2, eph->file) != 2)
                {
                    fatalerror("writeph_seq_write: can't write the segment directory at record %d word %d of "
                               "the ephemeris file '%s'\nSystem error : '%s'\n",
                               writeph_word2rec(eph->header.free), eph->header.free, eph->header.ifname,
                               calceph_strerror_errno(buffer_error));
                    return 0;
                }
                eph->header.free += 2;
                break;
            }
        default:
            fatalerror("writeph_seq_write: unsupported segment type %d\n", data_type);
            return 0;
    }

    /* pad the current record if necessary */
    if (!writeph_padrec(eph, '\0', 0))
        return 0;

    /* update the header of the ephemeris */
    if (!writeph_header(eph, 0))
        return 0;

    return 1;
}

/*---------------------------------------------------------------------------*/
/*!
    reserve a segment of any type to a spk file in parallel mode.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
    @param data_type (in) segment type
    @param target_count (in) number of targets
    @param targets (in) array of target ids
    @param center (in) center id
    @param frame (in) reference frame id
    @param start_jd0 (in) start time of the segments, in days (JD0 part)
    @param start_frac (in) start time of the segments, in days (fractional part)
    @param end_jd0 (in) end time of the segments, in days (JD0 part)
    @param end_frac (in) end time of the segments, in days (fractional part)
    @param intlens_jd (in) array of interpolation interval lengths of each segment,
                               in days (for fixed time step, unused otherwise)
    @param record_counts (in) array of number of records in each segment
    @param deg (in) degree of the polynomials
    @param segids (in) array of segment identifiers
*/
/*---------------------------------------------------------------------------*/
int writeph_par_reserve
    (t_writephbin * eph,
     enum SPICEfiletype file_type,
     enum SPKdatatype data_type,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0,
     double start_frac,
     double end_jd0, double end_frac, const double *intlens_jd, const int *record_counts, int deg, const char *segids[])
{
    buffer_error_t buffer_error;

    if (!eph)
    {
        fatalerror("writeph_par_reserve: wrong parameter, eph is NULL\n");
        return 0;
    }

    /* check if a generic segment is being written */
    if (eph->gen_segs != NULL)
    {
        fatalerror
            ("writeph_par_reserve: a generic segment is being written, you must end it before writing another segment\n");
        return 0;
    }

    /* check if there is enough space to reserve the segments */
    if (!writeph_checksize(data_type, eph->header.free, target_count, record_counts, deg))
    {
        fatalerror("writeph_par_reserve: not enough space in the ephemeris file '%s' to reserve %d segments\n",
                   eph->header.ifname, target_count);
        return 0;
    }

    int jd2000 = 2451545;
    double start_sec = ((start_jd0 - jd2000) + start_frac) * 86400;
    double end_sec = ((end_jd0 - jd2000) + end_frac) * 86400;

    /* convert TCB to TDB if needed */
    double start_sec_tdb = start_sec;
    double end_sec_tdb = end_sec;

    if (data_type == SPK_SEGTYPE102 || data_type == SPK_SEGTYPE103 || data_type == SPK_SEGTYPE120)
    {
        start_sec_tdb = writeph_tcb2tdb(start_sec);
        end_sec_tdb = writeph_tcb2tdb(end_sec);
    }

    /* allocate memory for the reservation descriptor */
    struct reservation *reserv = (struct reservation *) calloc(1, sizeof(struct reservation));

    if (reserv == NULL)
    {
        fatalerror("writeph_par_reserve: memory allocation error\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        return 0;
    }

    /* initialize the reservation descriptor */
    if (eph->reservations)
        reserv->id = eph->reservations->id + 1;
    else
        reserv->id = 1;
    reserv->target_count = target_count;
    reserv->start_sec = start_sec;
    reserv->end_sec = end_sec;
    reserv->center = center;
    reserv->frame = frame;
    reserv->data_type = (int) data_type;
    reserv->degree = deg;
    switch (reserv->data_type)
    {
        case SPK_SEGTYPE2:
        case SPK_SEGTYPE102:
            reserv->rsize = 2 + 3 * (reserv->degree + 1);
            break;
        case SPK_SEGTYPE3:
        case SPK_SEGTYPE103:
            reserv->rsize = 2 + 6 * (reserv->degree + 1);
            break;
        case SPK_SEGTYPE8:
        case SPK_SEGTYPE12:
        case SPK_SEGTYPE9:
        case SPK_SEGTYPE13:
            reserv->rsize = 6;
            break;
        default:
            fatalerror("writeph_par_write: unsupported segment type %d\n", reserv->data_type);
            return 0;
    }
    reserv->trajectories = malloc(target_count * sizeof(struct trajectory));
    if (reserv->trajectories == NULL)
    {
        fatalerror("writeph_par_reserve: memory allocation error\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        free(reserv);
        return 0;
    }
    reserv->next = NULL;

    int j;

    for (j = 0; j < target_count; j++)
    {
        double intlen_sec = 0.0;

        if (intlens_jd != NULL)
            intlen_sec = intlens_jd[j] * 86400;

        if (record_counts[j] <= 0 || (intlens_jd != NULL && intlen_sec <= 0.0) || segids[j] == NULL)
        {
            fatalerror("writeph_par_reserve: wrong parameter, n[%d]=%d, intlens_jd[%d]=%f, segids[%d]=%s\n",
                       j, record_counts[j], j, intlen_sec, j, segids[j]);
            free(reserv->trajectories);
            free(reserv);
            return 0;
        }

        /* initialize the segment */
        int segment_size =
            writeph_init(file_type, start_sec_tdb, end_sec_tdb, targets[j], center, frame, data_type, record_counts[j],
                         deg, segids[j], eph);

        if (segment_size == 0)
        {
            free(reserv->trajectories);
            free(reserv);
            return 0;
        }

        /* set the segment bounds */
        int first = writeph_rec2word(writeph_curr(eph));
        int last = first + segment_size - 1;

        /* write the segment metadata */
        switch (data_type)
        {
            case SPK_SEGTYPE2:
            case SPK_SEGTYPE3:
            case SPK_SEGTYPE8:
            case SPK_SEGTYPE12:
            case SPK_SEGTYPE102:
            case SPK_SEGTYPE103:
                {

                    double third_value;

                    if (data_type == SPK_SEGTYPE2 || data_type == SPK_SEGTYPE3 || data_type == SPK_SEGTYPE102 ||
                        data_type == SPK_SEGTYPE103)
                        third_value =
                            (double) ((data_type == SPK_SEGTYPE2 ||
                                       data_type == SPK_SEGTYPE102 ? 3 : 6) * (deg + 1) + 2);
                    else if (data_type == SPK_SEGTYPE8)
                        third_value = (double) deg;
                    else
                        third_value = (double) ((deg + 1) / 2 - 1);

                    double directory[4] = { start_sec, intlen_sec, third_value, (double) record_counts[j] };

                    /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
                    off_t offset = ((off_t) last - 4) * (off_t) sizeof(double);

                    if (!writeph_pwrite(eph, directory, 4 * sizeof(double), offset, 0))
                    {
                        free(reserv->trajectories);
                        free(reserv);
                        return 0;
                    }
                    break;
                }
            case SPK_SEGTYPE9:
            case SPK_SEGTYPE13:
                {
                    double directory[2] = { (double) deg, (double) record_counts[j] };

                    /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
                    off_t offset = ((off_t) last - 2) * (off_t) sizeof(double);

                    if (!writeph_pwrite(eph, directory, 2 * sizeof(double), offset, 0))
                    {
                        free(reserv->trajectories);
                        free(reserv);
                        return 0;
                    }
                    break;
                }
            default:
                fatalerror("writeph_seq_write: unsupported segment type %d\n", data_type);
                return 0;
        }

        /* update the first free word pointer */
        eph->header.free = last + 1;

        /* pad the current record if necessary */
        if (!writeph_padrec(eph, '\0', 0))
        {
            free(reserv->trajectories);
            free(reserv);
            return 0;
        }

        /* allocate memory for the trajectory descriptor */
        struct trajectory traj;

        /* initialize the trajectory descriptor */
        traj.first = first;
        traj.record_count = record_counts[j];
        traj.intlen_sec = intlen_sec;
        /* add the new trajectory descriptor into the reservation */
        reserv->trajectories[j] = traj;
    }

    /* update the header of the ephemeris */
    if (!writeph_header(eph, 0))
    {
        free(reserv->trajectories);
        free(reserv);
        return 0;
    }

    /* link the new reservation descriptor into the ephemeris */
    reserv->next = eph->reservations;
    eph->reservations = reserv;

    return reserv->id;
}

/*---------------------------------------------------------------------------*/
/*!
    write data into a reserved segment of any type to a spk file in parallel mode.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
    @param data_type (in) segment type
    @param reservation (in) reservation id
    @param target_index (in) index of the target in the reservation
    @param record_begin_index (in) index of the first record to write
    @param record_count (in) number of records to write
    @param data (in) array of data to write, must contain record_count * record_size values
    @param epochs (in) epochs to write (for unequal time steps, unused otherwise), must contain record_count values
*/
/*---------------------------------------------------------------------------*/
int writeph_par_write
    (t_writephbin * eph,
     enum SPKdatatype data_type,
     int reservation,
     int target_index, int record_begin_index, int record_count, const double *data, const double *epochs)
{
    if (!eph)
    {
        fatalerror("writeph_par_write: wrong parameter, eph is NULL\n");
        return 0;
    }

    struct reservation *reserv = eph->reservations;
    struct trajectory traj;

    /* find the reservation */
    while (reserv != NULL && reserv->id != reservation)
        reserv = reserv->next;
    if (reserv == NULL)
    {
        fatalerror("writeph_par_write: reservation %d not found\n", reservation);
        return 0;
    }
    /* check for segment type consistency */
    if (reserv->data_type != (int) data_type)
    {
        fatalerror("writeph_par_write: reservation %d has type %d, requested type is %d\n", reservation,
                   reserv->data_type, (int) data_type);
        return 0;
    }
    /* check target index */
    if (target_index < 0 || target_index >= reserv->target_count)
    {
        fatalerror("writeph_par_write: wrong parameter, target_index %d out of range [0,%d[\n", target_index,
                   reserv->target_count);
        return 0;
    }

    /* find the trajectory */
    traj = reserv->trajectories[target_index];

    /* check record count */
    if (record_count < 0)
    {
        fatalerror("writeph_par_write: wrong parameter, record_count %d must be positive\n", record_count);
        return 0;
    }
    /* check record range */
    if (record_begin_index < 0 || record_begin_index + record_count > traj.record_count)
    {
        fatalerror("writeph_par_write: wrong parameter, record range [%d,%d[ out of range [0,%d[\n", record_begin_index,
                   record_begin_index + record_count, reserv->trajectories[target_index].record_count);
        return 0;
    }

    /* prepare the records (heap allocated) */
    double *records = malloc((size_t) record_count * reserv->rsize * sizeof(double));

    if (!records)
    {
        fatalerror("out of memory\n");
        return 0;
    }

    /* macro for 2D indexing */
#define RECORD(j, i) records[(size_t)(j) * reserv->rsize + (i)]

    switch (reserv->data_type)
    {
        case SPK_SEGTYPE2:
        case SPK_SEGTYPE3:
        case SPK_SEGTYPE102:
        case SPK_SEGTYPE103:
            {
                int j;
                int set_size = reserv->rsize - 2;
                int radius = traj.intlen_sec / 2;

                for (j = 0; j < record_count; j++)
                {
                    RECORD(j, 0) = reserv->start_sec + (record_begin_index + j + 0.5) * traj.intlen_sec;
                    RECORD(j, 1) = radius;

                    memcpy(&RECORD(j, 2), data + (size_t) j * set_size, (size_t) set_size * sizeof(double));
                }
                break;
            }

        case SPK_SEGTYPE8:
        case SPK_SEGTYPE12:
        case SPK_SEGTYPE9:
        case SPK_SEGTYPE13:
            memcpy(records, data, (size_t) reserv->rsize * record_count * sizeof(double));
            break;

        default:
            free(records);
            fatalerror("writeph_par_write: unsupported segment type %d\n", reserv->data_type);
            return 0;
    }

    /* compute count and offset */
    size_t count = reserv->rsize * (size_t) record_count * sizeof(double);
    off_t offset = (traj.first + (off_t) record_begin_index * reserv->rsize - 1) * (off_t) sizeof(double);

    /* write the records */
    if (!writeph_pwrite(eph, records, count, offset, 1))
    {
        free(records);
        return 0;
    }

    free(records);

    /* for unequal time step segments */
    if (reserv->data_type == SPK_SEGTYPE9 || reserv->data_type == SPK_SEGTYPE13)
    {
        /* write the epochs and directories records */
        int j;

        for (j = 0; j < record_count; j++)
        {
            /* convert epoch */
            double epoch_sec = (epochs[j] - 2451545.0) * 86400.0;

            /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
            off_t epoch_offset =
                (traj.first + (off_t) traj.record_count * 6 + record_begin_index + j - 1) * (off_t) sizeof(double);

            /* write the epoch */
            if (!writeph_pwrite(eph, &epoch_sec, sizeof(double), epoch_offset, 1))
                return 0;

            /* write a directory record if needed */
            if ((record_begin_index + j + 1) % 100 == 0)
            {
                /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
                off_t dir_offset =
                    (traj.first + (off_t) traj.record_count * 6 + traj.record_count + (record_begin_index + j) / 100 -
                     1) * (off_t) sizeof(double);

                if (!writeph_pwrite(eph, &epoch_sec, sizeof(double), dir_offset, 1))
                    return 0;
            }
        }
    }
    return 1;
}

/*---------------------------------------------------------------------------*/
/*!
    begin writing a generic segment to a spk file.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
    @param data_type (in) segment type
    @param target (in) target id
    @param center (in) center id
    @param frame (in) reference frame id
    @param start_jd0 (in) start time of the segment in days (JD0 part)
    @param start_frac (in) start time of the segment in days (fractional part)
    @param end_jd0 (in) end time of the segment in days (JD0 part)
    @param end_frac (in) end time of the segment in days (fractional part)
    @param deg (in) degree of the polynomials
    @param segid (in) segment identifier
*/
/*---------------------------------------------------------------------------*/
int writeph_spk_begin(t_writephbin *eph, enum SPKdatatype data_type, int target, int center, int frame,
                      double start_jd0, double start_frac, double end_jd0, double end_frac, int deg, const char *segid)
{
    buffer_error_t buffer_error;

    if (!eph)
    {
        fatalerror("writeph_spk_begin: wrong parameter, eph is NULL\n");
        return 0;
    }

    if (eph->gen_segs != NULL)
    {
        fatalerror
            ("writeph_spk_begin: a generic segment is already being written, you must end it before starting another segment\n");
        return 0;
    }

    /* for now only type 14 is supported */
    if (data_type != SPK_SEGTYPE14)
    {
        fatalerror("writeph_spk_begin: unsupported segment type %d, only type %d is supported\n", data_type,
                   SPK_SEGTYPE14);
        return 0;
    }

    /* allocate memory for the generic segment */
    struct generic_segment *generic_segment = calloc(1, sizeof(struct generic_segment));

    if (!generic_segment)
    {
        fatalerror("writeph_spk_begin: failed to allocate memory for generic segment\n");
        return 0;
    }

    /* convert times to seconds from J2000 */
    int jd2000 = 2451545;
    double start_sec_tdb = ((start_jd0 - jd2000) + start_frac) * 86400;
    double end_sec_tdb = ((end_jd0 - jd2000) + end_frac) * 86400;

    /* the generic segment must be set before initialization so its *seg_des_bound_fields_address* will be set */
    eph->gen_segs = generic_segment;

    /* initialize the segment */
    if (!writeph_init(DAF_SPK, start_sec_tdb, end_sec_tdb, target, center, frame, data_type, 0, deg, segid, eph))
    {
        free(generic_segment);
        return 0;
    }

    /* write the constant partition */

    /* this is because the constant is actually supposed to contain the number of coefficients and
       and not the degree of the polynom, this is wrongly documented in the SPICE SPK Required reading documentation */
    double deg_d = (double) (deg + 1);

    if (fwrite(&deg_d, sizeof(double), 1, eph->file) != 1)
    {
        fatalerror("writeph_spk_begin: can't write the constant partition at record %d word %d of "
                   "the ephemeris file '%s'\nSystem error : '%s'\n",
                   writeph_word2rec(eph->header.free), eph->header.free, eph->header.ifname,
                   calceph_strerror_errno(buffer_error));
        free(generic_segment);
        return 0;
    }
    eph->header.free += 1;

    generic_segment->epochs = malloc(PARTITIONS_DEFAULT_CAPACITY * sizeof(double));
    if (!generic_segment->epochs)
    {
        fatalerror("writeph_spk_begin: failed to allocate memory for epochs\n");
        free(generic_segment);
        return 0;
    }
    generic_segment->epochs_count = 0;
    generic_segment->epochs_capacity = PARTITIONS_DEFAULT_CAPACITY;
    generic_segment->begin_address = eph->header.free - 1;
    generic_segment->deg = deg;

    /* initialize the metadata */
    generic_segment->meta.conbas = 0;
    generic_segment->meta.ncon = 1;
    generic_segment->meta.pktbas = 1;
    generic_segment->meta.npkt = 0;
    generic_segment->meta.pktsz = 2 + 6 * (deg + 1);
    generic_segment->meta.nmeta = 17;
    /* unused */
    generic_segment->meta.pdrbas = -1;
    generic_segment->meta.npdr = 0;
    generic_segment->meta.rsvbas = -1;
    generic_segment->meta.nrsv = 0;

    return 1;
}

/*---------------------------------------------------------------------------*/
/*!
    add data into a generic segment to a spk file.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
    @param record_count (in) number of records to add
    @param data (in) array of data to write
    @param epochs (in) epochs to write, must contain record_count values
*/
/*---------------------------------------------------------------------------*/
int writeph_spk_add(t_writephbin *eph, int record_count, const double *data, const double *epochs)
{
    struct generic_segment *generic_segment = eph->gen_segs;

    if (!generic_segment)
    {
        fatalerror("writeph_spk_add: no generic segment initialized, you must call writeph_spk_begin first\n");
        return 0;
    }

    /* check if there is enough space in the ephemeris file */
    if (!writeph_checksize(SPK_SEGTYPE14, eph->header.free, 1, &record_count, generic_segment->deg))
    {
        fatalerror("writeph_spk_add: not enough space in the ephemeris file '%s' to write %d records\n",
                   eph->header.ifname, record_count);
        return 0;
    }

    /* check if we need to reallocate memory */
    if (generic_segment->epochs_count + record_count > generic_segment->epochs_capacity)
    {
        int new_capacity = generic_segment->epochs_capacity * 1.5;

        while (generic_segment->epochs_count + record_count > new_capacity)
            new_capacity *= 1.5;

        double *new_epochs = realloc(generic_segment->epochs, new_capacity * sizeof(double));

        if (!new_epochs)
        {
            fatalerror("writeph_spk_add: failed to reallocate memory for epochs\n");
            return 0;
        }
        generic_segment->epochs = new_epochs;
        generic_segment->epochs_capacity = new_capacity;
    }

    /* store the epochs in the generic segment structure */
    memcpy(generic_segment->epochs + generic_segment->epochs_count, epochs, record_count * sizeof(double));
    generic_segment->epochs_count += record_count;

    int rsize = 2 + 6 * (generic_segment->deg + 1);

    /* directly write the packets */
    if (writeph_pwrite
        (eph, data, record_count * rsize * sizeof(double), (eph->header.free - 1) * (off_t) sizeof(double), 1) == 0)
        return 0;
    eph->header.free += record_count * rsize;

    /* update the metadata */
    generic_segment->meta.npkt += record_count;
    generic_segment->meta.nref += record_count;

    return 1;
}

/*---------------------------------------------------------------------------*/
/*!
    end writing a generic segment to a spk file.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
*/
/*---------------------------------------------------------------------------*/
int writeph_spk_end(t_writephbin *eph)
{
    buffer_error_t buffer_error;

    if (!eph)
    {
        fatalerror("writeph_spk_end: wrong parameter, eph is NULL\n");
        return 0;
    }

    struct generic_segment *generic_segment = eph->gen_segs;

    if (!generic_segment)
    {
        fatalerror("writeph_spk_end: no generic segment initialized, you must call writeph_spk_begin first\n");
        return 0;
    }

    /* convert all the epochs from Julian days to seconds past J2000 */
    int j;
    double *epochs_sec = malloc(generic_segment->epochs_count * sizeof(double));

    if (!epochs_sec)
    {
        fatalerror("writeph_par_reserve: memory allocation error\nSystem error : '%s'\n",
                   calceph_strerror_errno(buffer_error));
        return 0;
    }
    for (j = 0; j < generic_segment->epochs_count; j++)
        epochs_sec[j] = (generic_segment->epochs[j] - 2451545.0) * 86400.0;
    /* write the epochs */
    if (!writeph_pwrite
        (eph, epochs_sec, generic_segment->epochs_count * sizeof(double),
         (eph->header.free - 1) * (off_t) sizeof(double), 0))
    {
        free(epochs_sec);
        return 0;
    }
    eph->header.free += generic_segment->epochs_count;

    /* write the epoch directory */
    int n_dir = writeph_ndir(generic_segment->epochs_count);

    for (j = 1; j <= n_dir; j++)
    {
        double epoch_sec = (epochs_sec[j * 100 - 1] - 2451545.0) * 86400.0;

        /* write the directory if needed */
        if ((j + 1) % 100 == 0)
        {
            /* compute the offset, make sure the expression is of type off_t to prevent an int overflow */
            off_t dir_offset = (eph->header.free + j / 100 - 1) * (off_t) sizeof(double);

            if (!writeph_pwrite(eph, &epoch_sec, sizeof(double), dir_offset, 0))
            {
                free(epochs_sec);
                return 0;
            }
        }
    }
    eph->header.free += n_dir;
    free(epochs_sec);

    /* update the metadata */
    generic_segment->meta.refbas =
        generic_segment->meta.pktbas + generic_segment->epochs_count * generic_segment->meta.pktsz;
    generic_segment->meta.rdrbas = n_dir == 0 ? 0 : generic_segment->meta.refbas + generic_segment->epochs_count;
    generic_segment->meta.nrdr = n_dir;

    /* write the metadata */
    if (!writeph_pwrite
        (eph, &generic_segment->meta, sizeof(struct meta_partition), (eph->header.free - 1) * (off_t) sizeof(double),
         0))
        return 0;
    eph->header.free += generic_segment->meta.nmeta;

    /* update the segment bounds */
    int bounds[2] = { generic_segment->begin_address, eph->header.free - 1 };
    if (!writeph_pwrite
        (eph, &bounds[0], 2 * sizeof(int), (generic_segment->seg_des_bound_fields_address) * sizeof(double), 0))
        return 0;

    free(generic_segment->epochs);
    free(generic_segment);
    eph->gen_segs = NULL;

    /* pad the segment if necessary */
    if (!writeph_padrec(eph, '\0', 0))
        return 0;

    /* update the header of the ephemeris */
    if (!writeph_header(eph, 0))
        return 0;

    return 1;
}
