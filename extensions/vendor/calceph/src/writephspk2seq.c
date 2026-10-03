/*-----------------------------------------------------------------*/
/*!
  \file writephspk2seq.c
  \brief perform the sequential writing of a type 2 segment to a spk file.

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
    write a type 2 segment to a spk file in sequential mode

    return 0 on error.
    return 1 on success.

  @param eph (in) ephemermis descriptor
  @param target (in) NAIF id of the target
  @param center (in) NAIF id of the center of the target
  @param frame (in) NAIF code of the reference frame
  @param start_jd0_tdb (in) start epoch (integer part) in TDB Julian days
  @param start_frac_tdb (in) start epoch (fractional part) in TDB Julian days
  @param end_jd0_tdb (in) end epoch (integer part) in TDB Julian days
  @param end_frac_tdb (in) end epoch (fractional part) in TDB Julian days
  @param intlen (in) length of the interpolation interval (in seconds)
  @param polynomials (in) array of coefficients, it should contain n*3*(deg+1)) values
  @param record_count (in) number of interpolation intervals
  @param deg (in) degree of the interpolating polynomial
  @param segid (in) segment identifier, it should be a string of 40 characters
*/
/*--------------------------------------------------------------------------*/

int writeph_spk2_seq_write
    (t_writephbin * eph,
     int target,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, double intlen_jd_tdb, const double *polynomials, int record_count, int deg, const char *segid)
{
    return writeph_seq_write(eph, DAF_SPK, SPK_SEGTYPE2, target, center, frame, start_jd0_tdb, start_frac_tdb,
                             end_jd0_tdb, end_frac_tdb, intlen_jd_tdb, polynomials, NULL, record_count, deg, segid);
}
