/*-----------------------------------------------------------------*/
/*!
  \file writephspk14seq.c
  \brief perform the sequential writing of a type 14 segment to a spk file.

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
    begin writing a type 14 segment to a spk file.

       return 0 on error.
       return 1 on success.

       @param eph (in) ephemeris descriptor
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
int writeph_spk14_begin(t_writephbin *eph, int target, int center, int frame,
                        double start_jd0, double start_frac, double end_jd0, double end_frac, int deg,
                        const char *segid)
{
    return writeph_spk_begin(eph, SPK_SEGTYPE14, target, center, frame,
                             start_jd0, start_frac, end_jd0, end_frac, deg, segid);
}

/*---------------------------------------------------------------------------*/
/*!
    add data to a type 14 segment to a spk file.

       return 0 on error.
       return 1 on success.

       @param eph (in) ephemeris descriptor
       @param data (in) array of coefficients, it should contain (n*6*(deg+1)) values
       @param epochs (in) array of epochs, it should contain record_count values
       @param record_count (in) number of interpolation intervals
*/
/*---------------------------------------------------------------------------*/
int writeph_spk14_add(t_writephbin *eph, int record_count, const double *data, const double *epochs)
{
    return writeph_spk_add(eph, record_count, data, epochs);
}

/*---------------------------------------------------------------------------*/
/*!
    end writing a type 14 segment to a spk file.

    return 0 on error.
    return 1 on success.

    @param eph (in) ephemeris descriptor
*/
/*---------------------------------------------------------------------------*/
int writeph_spk14_end(t_writephbin *eph)
{
    return writeph_spk_end(eph);
}
