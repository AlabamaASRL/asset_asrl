/*-----------------------------------------------------------------*/
/*!
  \file f90calcephgetfov.c
  \brief Fortran 77/90/95 interface for the getfov functions.

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright, 2025-2026,CNRS
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

#include "f90calcephinternal.h"
#include "calcephconfigfortran.h"
#include "f90fillspace.h"

int FC_GLOBAL_(f90calceph_getfov, F90CALCEPH_GETFOV) (long long *eph, int *instrumentid, int *shape,
                                                      t_calcephcharvalue frame, double *vector,
                                                      double *arraybounds, int *nbounds);

int FC_GLOBAL_(f90calceph_getfov, F90CALCEPH_GETFOV) (long long *eph, int *instrumentid, int *shape,
                                                      t_calcephcharvalue frame, double *vector,
                                                      double *arraybounds, int *nbounds)
{
    int ret = calceph_getfov(f90calceph_getaddresseph(eph), *instrumentid, shape, frame, vector,
                             *nbounds == 0 ? NULL : arraybounds, *nbounds);

    if (ret > 0)
        calceph_fortranfillspace(frame, CALCEPH_MAX_CONSTANTVALUE);
    return ret;
}
