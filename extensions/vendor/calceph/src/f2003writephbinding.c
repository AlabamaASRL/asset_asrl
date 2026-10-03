/*-----------------------------------------------------------------*/
/*!
  \file f2003writephbinding.c
  \brief Fortran 2003 interface for writeph : C binding.

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright, 2026,CNRS
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

#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"

#include "f2003writephbinding.h"
#include "real.h"
#include "util.h"
#include "calcephspice.h"

/*--------------------------------------------------------------------------*/
/*!
   map the Fortran array of segment name to the C array

   @param funcname (in) name of the function
   @param count (in) number of segments
   @param fortran_segids (in) name of the segments - fortran
   @param c_segids (out) name of the segments - c
   @param c_segids_storage (out) name of the segments - c internal storage
   return 0 on error, otherwise 1.
*/
/*--------------------------------------------------------------------------*/
static int map_segment_name_fortran_to_c(const char *funcname, int count, const char *fortran_segids,
                                         char ***c_segids, char **c_segids_storage)
{
    const int max_segmentation_len = SEGMENTID_LEN;
    int k, l;
    char **pc_segids;
    char *pc_segids_storage;

    *c_segids = NULL;
    *c_segids_storage = NULL;

    *c_segids = pc_segids = (char **) malloc(sizeof(char *) * (count));
    *c_segids_storage = pc_segids_storage = (char *) malloc(sizeof(char) * (max_segmentation_len + 1) * (count));

    if (pc_segids != NULL && pc_segids_storage != NULL)
    {
        for (k = 0; k < count; k++)
        {
            pc_segids[k] = pc_segids_storage + (max_segmentation_len + 1) * k;
            memcpy(pc_segids[k], fortran_segids + max_segmentation_len * k, max_segmentation_len * sizeof(char));
            pc_segids[k][max_segmentation_len] = '\0';
            l = max_segmentation_len - 1;
            while (l > 0 && pc_segids[k][l] == ' ')
            {
                pc_segids[k][l] = '\0';
                l--;
            }
        }
        return 1;
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for %s\nSystem error : '%s'\n", funcname,
                   calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */

    return 0;
}

/*--------------------------------------------------------------------------*/
/*!
   free memory allocated by map_segment_name_fortran_to_c
   @param c_segids (inout) name of the segments - c
   @param c_segids_storage (inout) nname of the segments - c internal storage
*/
/*--------------------------------------------------------------------------*/
static void map_segment_release(char **c_segids, char *c_segids_storage)
{
    if (c_segids != NULL)
        free(c_segids);
    if (c_segids_storage != NULL)
        free(c_segids_storage);
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 2 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int f2003writeph_spk2_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb,
                                  const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk2_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_spk2_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb,
                                       record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 3 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int f2003writeph_spk3_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb,
                                  const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk3_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_spk3_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb,
                                       record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 8 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the stateials (same for all segments)
   @param segids array of segment identifiers - fortran array

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_spk8_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb,
                                  const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk9_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_spk8_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb,
                                       record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 9 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_spk9_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const int *record_counts,
                                  int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk9_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_spk9_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, record_counts, deg,
                                       (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 12 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the stateials (same for all segments)
   @param segids array of segment identifiers

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_spk12_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk12_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_spk12_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                        start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts, deg,
                                        (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 13 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int f2003writeph_spk13_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb, double end_frac_tdb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk13_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_spk13_par_reserve(eph, target_count, targets, center, frame, start_jd0_tdb,
                                        start_frac_tdb, end_jd0_tdb, end_frac_tdb, record_counts, deg,
                                        (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 102 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tcb start time of the segments, in days (JD0 part)
   @param start_frac_tcb start time of the segments, in days (fractional part)
   @param end_jd0_tcb end time of the segments, in days (JD0 part)
   @param end_frac_tcb end time of the segments, in days (fractional part
   @param intlens_jd_tcb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_spk102_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                    int frame, double start_jd0_tcb, double start_frac_tcb,
                                    double end_jd0_tcb, double end_frac_tcb,
                                    const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk102_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_spk102_par_reserve(eph, target_count, targets, center, frame, start_jd0_tcb,
                                         start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb,
                                         record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 103 segments to a spk file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tcb start time of the segments, in days (JD0 part)
   @param start_frac_tcb start time of the segments, in days (fractional part)
   @param end_jd0_tcb end time of the segments, in days (JD0 part)
   @param end_frac_tcb end time of the segments, in days (fractional part
   @param intlens_jd_tcb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_spk103_par_reserve(t_writephbin *eph, int target_count, const int *targets, int center,
                                    int frame, double start_jd0_tcb, double start_frac_tcb,
                                    double end_jd0_tcb, double end_frac_tcb,
                                    const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_spk103_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_spk103_par_reserve(eph, target_count, targets, center, frame, start_jd0_tcb,
                                         start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb,
                                         record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 2 segments to a pck file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part)
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int f2003writeph_pck2_par_reserve(t_writephbin *eph, int target_count, const int *targets,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb,
                                  const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_pck2_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_pck2_par_reserve(eph, target_count, targets, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb,
                                       record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 3 segments to a pck file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part)
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int f2003writeph_pck3_par_reserve(t_writephbin *eph, int target_count, const int *targets,
                                  int frame, double start_jd0_tdb, double start_frac_tdb,
                                  double end_jd0_tdb, double end_frac_tdb, const double *intlens_jd_tdb,
                                  const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_pck3_par_reserve", target_count, segids, &c_segids, &c_segids_storage);

    if (ret)
        ret = writeph_pck3_par_reserve(eph, target_count, targets, frame, start_jd0_tdb,
                                       start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb,
                                       record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 102 segments to a pck file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tcb start time of the segments, in days (JD0 part)
   @param start_frac_tcb start time of the segments, in days (fractional part)
   @param end_jd0_tcb end time of the segments, in days (JD0 part)
   @param end_frac_tcb end time of the segments, in days (fractional part)
   @param intlens_jd_tcb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_pck102_par_reserve(t_writephbin *eph, int target_count, const int *targets,
                                    int frame, double start_jd0_tcb, double start_frac_tcb,
                                    double end_jd0_tcb, double end_frac_tcb,
                                    const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_pck102_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_pck102_par_reserve(eph, target_count, targets, frame, start_jd0_tcb,
                                         start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb,
                                         record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}

/*--------------------------------------------------------------------------*/
/*!
   reserve space for several type 103 segments to a pck file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param center center id of all segments
   @param frame reference frame id of all segments
   @param start_jd0_tcb start time of the segments, in days (JD0 part)
   @param start_frac_tcb start time of the segments, in days (fractional part)
   @param end_jd0_tcb end time of the segments, in days (JD0 part)
   @param end_frac_tcb end time of the segments, in days (fractional part)
   @param intlens_jd_tcb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers (fortran array)

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/
int f2003writeph_pck103_par_reserve(t_writephbin *eph, int target_count, const int *targets,
                                    int frame, double start_jd0_tcb, double start_frac_tcb,
                                    double end_jd0_tcb, double end_frac_tcb,
                                    const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids)
{
    char **c_segids;
    char *c_segids_storage;
    int ret;

    ret = map_segment_name_fortran_to_c("writeph_pck103_par_reserve", target_count, segids, &c_segids,
                                        &c_segids_storage);

    if (ret)
        ret = writeph_pck103_par_reserve(eph, target_count, targets, frame, start_jd0_tcb,
                                         start_frac_tcb, end_jd0_tcb, end_frac_tcb, intlens_jd_tcb,
                                         record_counts, deg, (const char **) c_segids);

    map_segment_release(c_segids, c_segids_storage);

    return ret;
}
