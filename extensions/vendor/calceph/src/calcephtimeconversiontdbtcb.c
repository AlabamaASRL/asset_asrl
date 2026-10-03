/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversiontdbtcb.c
  \brief functions that compute tcb <-> tdb conversions

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris.

   Copyright, 2026, CNRS
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

#include <float.h>
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
#include "real.h"
#include "util.h"
#include "calcephinternal.h"

/*

 The constants come from the IAU 2006 Resolution B3
 Resolution B3: "Re-definition of Barycentric Dynamical Time, TDB"

The implementation is done using the equations 22 (same as IAU) of
@article{Turyshev_2025,
doi = {10.3847/1538-4357/adcc18},
url = {https://dx.doi.org/10.3847/1538-4357/adcc18},
year = {2025},
month = {may},
publisher = {The American Astronomical Society},
volume = {985},
number = {1},
pages = {140},
author = {Turyshev, Slava G. and Williams, James G. and Boggs, Dale H. and Park, Ryan S.},
title = {Relativistic Time Transformations between the Solar System Barycenter, Earth, and Moon},
journal = {The Astrophysical Journal},
}

or same (page 17) in
@ARTICLE{2010ITN....36....1P,
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

*/

#define LB_TCB2TDB +1.550519768E-08
#define T0 2443144.5003725
#define TDB0 (-6.55e-5 / 86400.0)

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TCB

   cf IAU 2006 resolution B3 : TDB = TCB - LB*(TCB-T0)*86400 + TDB0
                            => TCB = TDB + (  LB*(TDB-T0)*86400 - TDB0 ) / (1-LB)

    @return 0 on error, otherwise non-zero value

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tcb     (out) integer part of Julian Date (TCB)
    @param jdfrac_tcb  (out) fractional part of Julian Date (TCB)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tdb_to_jd_tcb(t_calcephbin *eph, double jd0_tdb, double jdfrac_tdb, double *jd0_tcb,
                                  double *jdfrac_tcb)
{
    /* GCOVR_EXCL_START */
    if (eph == NULL || jd0_tcb == NULL || jdfrac_tcb == NULL)
    {
        fatalerror("calceph_time_jd_tdb_to_jd_tcb: invalid input (eph=%p, jd0_tcb=%p, jdfrac_tcb=%p)\n", eph,
                   jd0_tcb, jdfrac_tcb);
        return 0;
    }
    /* GCOVR_EXCL_STOP */

    *jd0_tcb = jd0_tdb;
    *jdfrac_tcb = jdfrac_tdb + (((jd0_tdb - T0) + jdfrac_tdb) * LB_TCB2TDB - TDB0) / (1 - LB_TCB2TDB);

    if (*jdfrac_tcb >= 1.)
    {
        *jd0_tcb += 1.;
        *jdfrac_tcb -= 1.;
    }
    else if (*jdfrac_tcb < 0.)
    {
        *jd0_tcb -= 1.;
        *jdfrac_tcb += 1.;
    }
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TCB to TDB

   cf IAU 2006 resolution B3 : TDB = TCB - LB*(TCB-T0)*86400 + TDB0
                            => TCB = TDB + (  LB*(TDB-T0)*86400 - TDB0 ) / (1-LB)

    @return 0 on error, otherwise non-zero value

    @param eph         (in)  ephemeris object
    @param jd0_tcb      (in)  integer part of Julian Date (TCB)
    @param jdfrac_tcb   (in)  fractional part of Julian Date (TCB)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tcb_to_jd_tdb(t_calcephbin *eph, double jd0_tcb, double jdfrac_tcb, double *jd0_tdb,
                                  double *jdfrac_tdb)
{
    /* GCOVR_EXCL_START */
    if (eph == NULL || jd0_tdb == NULL || jdfrac_tdb == NULL)
    {
        fatalerror("calceph_time_jd_tcb_to_jd_tdb: invalid input (eph=%p, jd0_tdb=%p, jdfrac_tdb=%p)\n", eph,
                   jd0_tdb, jdfrac_tdb);
        return 0;
    }
    /* GCOVR_EXCL_STOP */

    *jd0_tdb = jd0_tcb;
    *jdfrac_tdb = jdfrac_tcb - LB_TCB2TDB * ((jd0_tcb - T0) + jdfrac_tcb) + TDB0;

    if (*jdfrac_tdb >= 1.)
    {
        *jd0_tdb += 1.;
        *jdfrac_tdb -= 1.;
    }
    else if (*jdfrac_tdb < 0.)
    {
        *jd0_tdb -= 1.;
        *jdfrac_tdb += 1.;
    }

    return 1;
}
