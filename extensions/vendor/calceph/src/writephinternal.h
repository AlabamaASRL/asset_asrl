/*-----------------------------------------------------------------*/
/*!
  \file writephinternal.h
  \brief private API for writeph library
         creates, open, and write to ephemeris files.

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
#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if HAVE_LIMITS_H
#include <limits.h>
#endif

#include "util.h"
#include "real.h"
#include "calceph.h"
#include "calcephspice.h"

#define WORD_PER_RECORD (int)(DAF_RECORD_LEN / sizeof(double))
#define PARTITIONS_DEFAULT_CAPACITY 1024

/*-----------------------------------------------------------------*/
/* structures */
/*-----------------------------------------------------------------*/

struct writephbin
{
    FILE *file;
    enum SPICEfiletype file_type;
    struct SPKHeader header;
    struct reservation *reservations;
    struct generic_segment *gen_segs;
    calceph_mtx_t lock;         /* lock for parallel writing */
};

struct segment_descriptor
{
    double d1;                  /* first double precision value */
    double d2;                  /* second double precision value */
    int i1;                     /* first integer value */
    int i2;                     /* ... */
    int i3;
    int i4;
    int i5;
    int i6;                     /* sixth integer value */
};

struct trajectory
{
    int first;                  /* index of the first record */
    int record_count;           /* number of records representing the trajectory */
    double intlen_sec;
};

struct reservation
{
    int id;
    int target_count;
    double start_sec;
    double end_sec;
    int center;
    int frame;
    int data_type;
    int degree;
    int rsize;
    struct trajectory *trajectories;
    struct reservation *next;
};

struct generic_segment
{
    int begin_address;
    double *epochs;
    int epochs_count;
    int epochs_capacity;
    int deg;
    int seg_des_bound_fields_address;
    struct meta_partition
    {
        double conbas;
        double ncon;
        double rdrbas;
        double nrdr;
        double rdrtyp;
        double refbas;
        double nref;
        double pdrbas;
        double npdr;
        double pdrtyp;
        double pktbas;
        double npkt;
        double rsvbas;
        double nrsv;
        double pktsz;
        double pktoff;
        double nmeta;
    } meta;
};

struct dump_header
{
    int location[2];
    double descriptor[3];
    int n_reservations;
};

/*-----------------------------------------------------------------*/
/* functions */
/*-----------------------------------------------------------------*/

/* write to a file at a given offset, using pwrite if available or fseeko/fwrite if not. */
int writeph_pwrite(t_writephbin * eph, const void *buf, size_t count, off_t offset, int para);

/* read from a file at a given offset, using pread if available or fseeko/fread if not. */
int writeph_pread(t_writephbin * eph, void *buf, size_t count, off_t offset, int para);

/* return the bff of the OS */
const char *writeph_getbff(void);

/* convert a record-based address into a word-based address */
int writeph_rec2word(int rec_addr);

/* convert a word-based address into a record-based address */
int writeph_word2rec(int word_addr);

/* return the current record address of the file */
int writeph_curr(t_writephbin * eph);

/* pad the current record with zeros if necessary */
int writeph_padrec(t_writephbin * eph, char pad, int para);

/* write in the first free record */
int writeph_rec(t_writephbin * eph, const void *ptr, size_t size, size_t nmemb, char pad, int para);

/* updates the bin file by rewriting its header */
int writeph_header(t_writephbin * eph, int para);

/* write a segment id to the first free record of the file */
int writeph_segid(t_writephbin * eph, const char *segid, int para);

/* add a new block of summary record to the doubly linked list */
int writeph_list(t_writephbin * eph, int para);

/* write a segment descriptor to a file */
int writeph_segdescriptor(t_writephbin * eph, struct segment_descriptor seg_des, int size, int para);

/* free all the reservations */
void writeph_free_reservations(struct reservation *reserv);

/* write a segment of any type to a spk file in sequential mode */
int writeph_seq_write
    (t_writephbin * eph,
     enum SPICEfiletype file_type,
     enum SPKdatatype data_type,
     int target,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb,
     double intlen_jd_tdb, const double *data, const double *epochs, int record_count, int deg, const char *segid);

/* reserve space for several segments of the same type to a spk file */
int writeph_par_reserve
    (t_writephbin * eph,
     enum SPICEfiletype file_type,
     enum SPKdatatype data_type,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[]);

/* write data into a reserved segment of any type to a spk file in parallel mode. */
int writeph_par_write
    (t_writephbin * eph,
     enum SPKdatatype data_type,
     int reservation,
     int target_index, int record_begin_index, int record_count, const double *data, const double *epochs);

/* begin writing a generic segment to a spk file */
int writeph_spk_begin
    (t_writephbin * eph,
     enum SPKdatatype type,
     int target,
     int center,
     int frame, double start_jd0, double start_frac, double end_jd0, double end_frac, int deg, const char *segid);

/* add data into a generic segment to a spk file */
int writeph_spk_add(t_writephbin * eph, int record_count, const double *data, const double *epochs);

/* end writing a generic segment to a spk file */
int writeph_spk_end(t_writephbin * eph);
