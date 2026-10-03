/*-----------------------------------------------------------------*/
/*!
  \file f2003writephbinding.h
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

/* reserve space for several type 2 segments to a spk file */
int f2003writeph_spk2_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 3 segments to a spk file */
int f2003writeph_spk3_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 8 segments to a spk file */
int f2003writeph_spk8_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 9 segments to a spk file */
int f2003writeph_spk9_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb, double end_frac_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 12 segments to a spk file */
int f2003writeph_spk12_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 13 segments to a spk file */
int f2003writeph_spk13_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb, double end_frac_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 102 segments to a spk file */
int f2003writeph_spk102_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tcb,
     double start_frac_tcb,
     double end_jd0_tcb,
     double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 103 segments to a spk file */
int f2003writeph_spk103_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int center,
     int frame,
     double start_jd0_tcb,
     double start_frac_tcb,
     double end_jd0_tcb,
     double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 2 segments to a pck file */
int f2003writeph_pck2_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 3 segments to a pck file */
int f2003writeph_pck3_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int frame,
     double start_jd0_tdb,
     double start_frac_tdb,
     double end_jd0_tdb,
     double end_frac_tdb, const double *intlens_jd_tdb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 102 segments to a pck file */
int f2003writeph_pck102_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int frame,
     double start_jd0_tcb,
     double start_frac_tcb,
     double end_jd0_tcb,
     double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids);

/* reserve space for several type 103 segments to a pck file */
int f2003writeph_pck103_par_reserve
    (t_writephbin * eph,
     int target_count,
     const int *targets,
     int frame,
     double start_jd0_tcb,
     double start_frac_tcb,
     double end_jd0_tcb,
     double end_frac_tcb, const double *intlens_jd_tcb, const int *record_counts, int deg, const char *segids);
