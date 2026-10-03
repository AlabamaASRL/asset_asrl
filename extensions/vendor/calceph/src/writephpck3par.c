/*-----------------------------------------------------------------*/
/*!
  \file writephpck3par.c
  \brief perform the parallel writing of a type 3 segment to a pck file.

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
   reserve space for several type 3 segments to a pck file in parallel mode

   @param eph pointer to the ephemeris structure
   @param target_count number of segments to reserve
   @param targets array of target ids
   @param frame reference frame id of all segments
   @param start_jd0_tdb start time of the segments, in days (JD0 part)
   @param start_frac_tdb start time of the segments, in days (fractional part)
   @param end_jd0_tdb end time of the segments, in days (JD0 part)
   @param end_frac_tdb end time of the segments, in days (fractional part
   @param intlens_jd_tdb array of interpolation interval lengths of each segment
   @param record_count array of number of lines of each segment
   @param deg degree of the polynomials (same for all segments)
   @param segids array of segment identifiers

   return the reservation id on success
   return 0 on error.
*/
/*--------------------------------------------------------------------------*/

int writeph_pck3_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids[])
{
    return writeph_par_reserve(eph, DAF_PCK, SPK_SEGTYPE3, target_count, targets, -1, frame,
                               start_jd0_tdb, start_frac_tdb, end_jd0_tdb, end_frac_tdb, intlens_jd_tdb, record_counts,
                               deg, segids);
}

/*--------------------------------------------------------------------------*/
/*! write data into a reserved segment of type 3 to a pck file in parallel mode
   @param eph pointer to the ephemeris structure
   @param reservation id of the reservation containing the segment
   @param target_index id of the trajectory/segment within the reservation
   @param record_begin_index line index within the segment
   @param record_count number of lines to write
   @param polynomials array of polynomial coefficients to write

   return 1 on success, 0 on error.
*/
/*--------------------------------------------------------------------------*/

int writeph_pck3_par_write(t_writephbin *eph, int reservation, int target_index, int record_begin_index,
                           int record_count, const double *polynomials)
{
    return writeph_par_write(eph, SPK_SEGTYPE3, reservation, target_index,
                             record_begin_index, record_count, polynomials, NULL);
}
