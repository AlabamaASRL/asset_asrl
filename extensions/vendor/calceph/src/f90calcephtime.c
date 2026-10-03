/*-----------------------------------------------------------------*/
/*!
  \file f90calcephtime.c
  \brief Fortran 77/90/95 interface for the time functions.

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
#include "f2003calcephbinding.h"

int FC_GLOBAL_(f90calceph_time_jd_to_cal,
               F90CALCEPH_TIME_JD_TO_CAL) (long long *eph, int *timescale, double *jd0, double *jdfrac,
                                           int *yy, int *month, int *day, int *hh, int *min, double *sec);

int FC_GLOBAL_(f90calceph_time_cal_to_jd,
               F90CALCEPH_TIME_CAL_TO_JD) (long long *eph, int *timescale, int *yy, int *month, int *day,
                                           int *hh, int *min, double *sec, double *jd0, double *jdfrac);

int FC_GLOBAL_(f90calceph_time_str_to_jd, F90CALCEPH_TIME_STR_TO_JD) (long long *eph, int *timescale,
                                                                      const char *str, double *jd0,
                                                                      double *jdfrac, long int len);

int FC_GLOBAL_(f90calceph_time_set_relationship_tt_tdb,
               F90CALCEPH_TIME_SET_RELATIONSHIP_TT_TDB) (long long *eph, int *model);

int FC_GLOBAL_(f90calceph_time_jd_tdb_to_jd_tt,
               F90CALCEPH_TIME_JD_TDB_TO_JD_TT) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                 double *jd0_tt, double *jdfrac_tt);

int FC_GLOBAL_(f90calceph_time_jd_tt_to_jd_tdb,
               F90CALCEPH_TIME_JD_TT_TO_JD_TDB) (long long *eph, double *jd0_tt, double *jdfrac_tt,
                                                 double *jd0_tdb, double *jdfrac_tdb);

int FC_GLOBAL_(f90calceph_time_jd_tdb_to_jd_tcb,
               F90CALCEPH_TIME_JD_TDB_TO_JD_TCB) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                  double *jd0_tcb, double *jdfrac_tcb);

int FC_GLOBAL_(f90calceph_time_jd_tcb_to_jd_tdb,
               F90CALCEPH_TIME_JD_TCB_TO_JD_TDB) (long long *eph, double *jd0_tcb, double *jdfrac_tcb,
                                                  double *jd0_tdb, double *jdfrac_tdb);

int FC_GLOBAL_(f90calceph_time_str_any_to_jd_tdb,
               F90CALCEPH_TIME_STR_ANY_TO_JD_TDB) (long long *eph, const char *str, double *jd0_tdb,
                                                   double *jdfrac_tdb, long int len);

int FC_GLOBAL_(f90calceph_time_str_utc_to_jd_tdb,
               F90CALCEPH_TIME_STR_UTC_TO_JD_TDB) (long long *eph, const char *str, double *jd0_tdb,
                                                   double *jdfrac_tdb, long int len);

int FC_GLOBAL_(f90calceph_time_cal_utc_to_jd_tdb,
               F90CALCEPH_TIME_CAL_UTC_TO_JD_TDB) (long long *eph, int *yy, int *month, int *day, int *hh,
                                                   int *min, double *sec, double *jd0_tdb, double *jdfrac_tdb);

int FC_GLOBAL_(f90calceph_time_jd_tdb_to_cal_utc,
               F90CALCEPH_TIME_JD_TDB_TO_CAL_UTC) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                   int *yy, int *month, int *day, int *hh, int *min, double *sec);

int FC_GLOBAL_(f90calceph_time_str_spacecraft_clock_to_jd_tdb,
               F90CALCEPH_TIME_STR_SPACECRAFT_CLOCK_TO_JD_TDB) (long long *eph, int *target, const char *str,
                                                                double *jd0_tdb, double *jdfrac_tdb, long int len);
int FC_GLOBAL_(f90calceph_time_jd_tdb_to_str_spacecraft_clock,
               F90CALCEPH_TIME_JD_TDB_TO_STR_SPACECRAFT_CLOCK) (long long *eph, int *target, double *jd0_tdb,
                                                                double *jdfrac_tdb, t_calcephcharvalue str);

/*--------------------------------------------------------------------------*/
/*! Dispatch Julian Date to calendar conversion according to timescale.

    Converts a two-part Julian Date into calendar date and time, using the
    appropriate timescale conversion:
      - CALCEPH_UTC → conversion with leap seconds (UTC)
      - Other timescales → continuous time (no leap seconds)

    @return 0 on error, otherwise non-zero value.

    @param eph       (in)  Pointer to CALCEPH binary ephemeris (for UTC only)
    @param timescale (in)  Timescale identifier (e.g., CALCEPH_UTC, CALCEPH_TT)
    @param jd0       (in)  Integer or main part of Julian Date
    @param jdfrac    (in)  Fractional part of Julian Date
    @param yy        (out) Pointer to integer year (Gregorian)
    @param month     (out) Pointer to integer month [1–12]
    @param day       (out) Pointer to integer day [1–31]
    @param hh        (out) Pointer to integer hour [0–23]
    @param min       (out) Pointer to integer minute [0–59]
    @param sec       (out) Pointer to floating-point seconds [0.0–61.0[
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_to_cal,
               F90CALCEPH_TIME_JD_TO_CAL) (long long *eph, int *timescale, double *jd0, double *jdfrac,
                                           int *yy, int *month, int *day, int *hh, int *min, double *sec)
{
    return calceph_time_jd_to_cal(f90calceph_getaddresseph(eph), *timescale, *jd0, *jdfrac, yy, month, day,
                                  hh, min, sec);
}

/*--------------------------------------------------------------------------*/
/*! Dispatch calendar-to-Julian-Date conversion according to timescale.

    Converts a Gregorian date and time into a Julian Date, selecting the
    correct conversion routine based on the specified timescale:
      - CALCEPH_UTC → uses UTC conversion (with leap seconds)
      - Other timescales → uses continuous timescale conversion (no leap seconds)

    @return 0 on error, otherwise non-zero value

    @param eph       (in)  Pointer to CALCEPH ephemeris structure
    @param timescale (in)  Timescale identifier (e.g., CALCEPH_UTC, CALCEPH_TT)
    @param yy        (in)  Calendar year
    @param month     (in)  Calendar month [1–12]
    @param day       (in)  Calendar day [1–31]
    @param hh        (in)  Hour [0–23]
    @param min       (in)  Minute [0–59]
    @param sec       (in)  Seconds [0.0–61.0[
    @param jd0       (out) Pointer to integer part of Julian Date
    @param jdfrac    (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_cal_to_jd,
               F90CALCEPH_TIME_CAL_TO_JD) (long long *eph, int *timescale, int *yy, int *month, int *day,
                                           int *hh, int *min, double *sec, double *jd0, double *jdfrac)
{
    return calceph_time_cal_to_jd(f90calceph_getaddresseph(eph), *timescale, *yy, *month, *day, *hh, *min, *sec,
                                  jd0, jdfrac);
}

/*--------------------------------------------------------------------------*/
/*! Parse and convert a time string to Julian Date.

    This function interprets a given time string in any supported format
    and converts it to a Julian Date.

    When the input timescale is 0 (undefined), the timescale inferred from
    the parsed time string is used.

    @return 0 on error, otherwise 1 on parsing or conversion failure

    @param eph       (in)  Pointer to CALCEPH ephemeris structure
    @param timescale (in)  Timescale identifier (e.g., CALCEPH_UTC, CALCEPH_TT)
    @param str       (in)  Input time string to be parsed
    @param jd0       (out) Pointer to integer part of Julian Date
    @param jdfrac    (out) Pointer to fractional part of Julian Date
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_str_to_jd, F90CALCEPH_TIME_STR_TO_JD) (long long *eph, int *timescale,
                                                                      const char *str, double *jd0,
                                                                      double *jdfrac, long int len)
{
    int res = 0;

    char *newname;

    newname = (char *) malloc(sizeof(char) * (len + 1));
    if (newname != NULL)
    {
        memcpy(newname, str, len * sizeof(char));
        newname[len] = '\0';
        res = calceph_time_str_to_jd(f90calceph_getaddresseph(eph), *timescale, newname, jd0, jdfrac);
        free(newname);
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for f90calceph_time_str_to_jd\nSystem "
                   "error : '%s'\n", calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */
    return res;
}

/*--------------------------------------------------------------------------*/
/*! set the relationship between TT and TDB time scales

    Two models are supported:
      - model = 0 : use of calceph_compute_unit()
      - model = 1 : use the default model based on the TLS file

    The function updates the internal field `relationship_tdb_tt` in the
    ephemeris structure.

    @return 0 on error, otherwise non-zero value.

    @param eph   (inout) ephemeris object
    @param model (in)    relationship model (0 or 1)
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_set_relationship_tt_tdb,
               F90CALCEPH_TIME_SET_RELATIONSHIP_TT_TDB) (long long *eph, int *model)
{
    return calceph_time_set_relationship_tt_tdb(f90calceph_getaddresseph(eph), *model);
}

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TT according to selected model

    @return 0 on error, otherwise non-zero value.

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tt     (out) integer part of Julian Date (TT)
    @param jdfrac_tt  (out) fractional part of Julian Date (TT)
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tdb_to_jd_tt,
               F90CALCEPH_TIME_JD_TDB_TO_JD_TT) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                 double *jd0_tt, double *jdfrac_tt)
{
    return calceph_time_jd_tdb_to_jd_tt(f90calceph_getaddresseph(eph), *jd0_tdb, *jdfrac_tdb, jd0_tt,
                                        jdfrac_tt);
}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TT to TDB according to the selected model
    defined in the ephemeris object.

    Depending on the value of eph->relationship_tdb_tt, either model 0 or model 1 is applied.

    @return 0 on error, otherwise non-zero value.

    @param eph         (in)  ephemeris object
    @param jd0_tt      (in)  integer part of Julian Date (TT)
    @param jdfrac_tt   (in)  fractional part of Julian Date (TT)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tt_to_jd_tdb,
               F90CALCEPH_TIME_JD_TT_TO_JD_TDB) (long long *eph, double *jd0_tt, double *jdfrac_tt,
                                                 double *jd0_tdb, double *jdfrac_tdb)
{
    return calceph_time_jd_tt_to_jd_tdb(f90calceph_getaddresseph(eph), *jd0_tt, *jdfrac_tt, jd0_tdb,
                                        jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! convert Julian Date from TDB to TCB according to selected model

    @return 0 on error, otherwise non-zero value.

    @param eph        (in)  ephemeris object
    @param jd0_tdb    (in)  integer part of Julian Date (TDB)
    @param jdfrac_tdb (in)  fractional part of Julian Date (TDB)
    @param jd0_tcb     (out) integer part of Julian Date (TCB)
    @param jdfrac_tcb  (out) fractional part of Julian Date (TCB)
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tdb_to_jd_tcb,
               F90CALCEPH_TIME_JD_TDB_TO_JD_TCB) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                  double *jd0_tcb, double *jdfrac_tcb)
{
    return calceph_time_jd_tdb_to_jd_tcb(f90calceph_getaddresseph(eph), *jd0_tdb, *jdfrac_tdb, jd0_tcb,
                                         jdfrac_tcb);
}

/*--------------------------------------------------------------------------*/
/*! Convert Julian Date from TCB to TDB according to the selected model
    defined in the ephemeris object.

    Depending on the value of eph->relationship_tdb_tcb, either model 0 or model 1 is applied.

    @return 0 on error, otherwise non-zero value.

    @param eph         (in)  ephemeris object
    @param jd0_tcb      (in)  integer part of Julian Date (TCB)
    @param jdfrac_tcb   (in)  fractional part of Julian Date (TCB)
    @param jd0_tdb     (out) integer part of Julian Date (TDB)
    @param jdfrac_tdb  (out) fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tcb_to_jd_tdb,
               F90CALCEPH_TIME_JD_TCB_TO_JD_TDB) (long long *eph, double *jd0_tcb, double *jdfrac_tcb,
                                                  double *jd0_tdb, double *jdfrac_tdb)
{
    return calceph_time_jd_tcb_to_jd_tdb(f90calceph_getaddresseph(eph), *jd0_tcb, *jdfrac_tcb, jd0_tdb,
                                         jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a time string to a Julian Date in the TDB timescale.

   This function parses the input string to determine the date and its
   original timescale (UTC, TAI, TT, TDB), then performs the necessary
   conversions using the ephemeris data to produce a TDB Julian Date.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)      Ephemeris descriptor
   @param str        (in)      Date/time string to parse
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_str_any_to_jd_tdb,
               F90CALCEPH_TIME_STR_ANY_TO_JD_TDB) (long long *eph, const char *str, double *jd0_tdb,
                                                   double *jdfrac_tdb, long int len)
{
    int res = 0;

    char *newname;

    newname = (char *) malloc(sizeof(char) * (len + 1));
    if (newname != NULL)
    {
        memcpy(newname, str, len * sizeof(char));
        newname[len] = '\0';
        res = calceph_time_str_any_to_jd_tdb(f90calceph_getaddresseph(eph), newname, jd0_tdb, jdfrac_tdb);
        free(newname);
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for f90calceph_time_str_any_to_jd_tdb\nSystem "
                   "error : '%s'\n", calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */
    return res;
}

/*--------------------------------------------------------------------------*/
/*! Convert a UTC time string to a Julian Date in the TDB timescale.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)      Ephemeris descriptor
   @param str        (in)      UTC time string to parse
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_str_utc_to_jd_tdb,
               F90CALCEPH_TIME_STR_UTC_TO_JD_TDB) (long long *eph, const char *str, double *jd0_tdb,
                                                   double *jdfrac_tdb, long int len)
{
    int res = 0;

    char *newname;

    newname = (char *) malloc(sizeof(char) * (len + 1));
    if (newname != NULL)
    {
        memcpy(newname, str, len * sizeof(char));
        newname[len] = '\0';
        res = calceph_time_str_utc_to_jd_tdb(f90calceph_getaddresseph(eph), newname, jd0_tdb, jdfrac_tdb);
        free(newname);
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for f90calceph_time_str_any_to_jd_tdb\nSystem "
                   "error : '%s'\n", calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */
    return res;
}

/*--------------------------------------------------------------------------*/
/*! Convert a UTC Calendar date to a Julian Date in the TDB timescale.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)      Ephemeris descriptor
   @param yy         (in)      Year
   @param month      (in)      Month
   @param day        (in)      Day
   @param hh         (in)      Hour
   @param min        (in)      Minute
   @param sec        (in)      Second
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_cal_utc_to_jd_tdb,
               F90CALCEPH_TIME_CAL_UTC_TO_JD_TDB) (long long *eph, int *yy, int *month, int *day, int *hh,
                                                   int *min, double *sec, double *jd0_tdb, double *jdfrac_tdb)
{
    return calceph_time_cal_utc_to_jd_tdb(f90calceph_getaddresseph(eph), *yy, *month, *day, *hh, *min,
                                          *sec, jd0_tdb, jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! Convert a TDB Julian Date to a UTC Calendar date.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)      Ephemeris descriptor
   @param jd0_tdb    (in)      Integer part of the TDB Julian Date
   @param jdfrac_tdb (in)      Fractional part of the TDB Julian Date
   @param yy         (out)     Year
   @param month      (out)     Month
   @param day        (out)     Day
   @param hh         (out)     Hour
   @param min        (out)     Minute
   @param sec        (out)     Second
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tdb_to_cal_utc,
               F90CALCEPH_TIME_JD_TDB_TO_CAL_UTC) (long long *eph, double *jd0_tdb, double *jdfrac_tdb,
                                                   int *yy, int *month, int *day, int *hh, int *min, double *sec)
{
    return calceph_time_jd_tdb_to_cal_utc(f90calceph_getaddresseph(eph), *jd0_tdb, *jdfrac_tdb, yy, month,
                                          day, hh, min, sec);
}

/*--------------------------------------------------------------------------*/
/*! convert the space clock time string to the Julian day expressed in TDB.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)      Ephemeris descriptor
   @param target      (in)     spaceraft id
   @param str        (in)      spacecraft clock time
   @param jd0_tdb    (out)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (out)     Fractional part of the resulting TDB Julian Date
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_str_spacecraft_clock_to_jd_tdb,
               F90CALCEPH_TIME_STR_SPACECRAFT_CLOCK_TO_JD_TDB) (long long *eph, int *target, const char *str,
                                                                double *jd0_tdb, double *jdfrac_tdb, long int len)
{
    int res = 0;

    char *newname;

    newname = (char *) malloc(sizeof(char) * (len + 1));
    if (newname != NULL)
    {
        memcpy(newname, str, len * sizeof(char));
        newname[len] = '\0';
        res =
            calceph_time_str_spacecraft_clock_to_jd_tdb(f90calceph_getaddresseph(eph), *target, newname, jd0_tdb,
                                                        jdfrac_tdb);
        free(newname);
    }
    /* GCOVR_EXCL_START */
    else
    {
        buffer_error_t buffer_error;

        fatalerror("Can't allocate memory for f90calceph_time_str_spacecraft_clock_to_jd_tdb\nSystem "
                   "error : '%s'\n", calceph_strerror_errno(buffer_error));
    }
    /* GCOVR_EXCL_STOP */
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the space clock time string to the Julian day expressed in TDB.

   @return 0 on error, otherwise non-zero value.

   @param eph        (in)     Ephemeris descriptor
   @param target     (in)     spaceraft id
   @param jd0_tdb    (in)     Integer part of the resulting TDB Julian Date
   @param jdfrac_tdb (in)     Fractional part of the resulting TDB Julian Date
   @param str        (out)    spacecraft clock time
 */
/*--------------------------------------------------------------------------*/
int FC_GLOBAL_(f90calceph_time_jd_tdb_to_str_spacecraft_clock,
               F90CALCEPH_TIME_JD_TDB_TO_STR_SPACECRAFT_CLOCK) (long long *eph, int *target,
                                                                double *jd0_tdb, double *jdfrac_tdb,
                                                                t_calcephcharvalue str)
{
    return f2003calceph_time_jd_tdb_to_str_spacecraft_clock(f90calceph_getaddresseph(eph), *target, *jd0_tdb,
                                                            *jdfrac_tdb, str);
}
