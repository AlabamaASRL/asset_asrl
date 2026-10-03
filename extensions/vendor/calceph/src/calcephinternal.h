/*-----------------------------------------------------------------*/
/*! 
  \file calcephinternal.h 
  \brief private API for calceph library
         access and interpolate INPOP and JPL Ephemeris data.

  \author  M. Gastineau 
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de Paris. 

   Copyright, 2008-2026, CNRS
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
knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
*/
/*-----------------------------------------------------------------*/

/*-----------------------------------------------------------------*/
/* private API */
/*-----------------------------------------------------------------*/

#include "calcephspice.h"
#include "calcephinpop.h"

/*-----------------------------------------------------------------*/
/* private API */
/*-----------------------------------------------------------------*/

/*-----------------------------------------------------------------*/
/* constants */
/*-----------------------------------------------------------------*/

/* index of the bodies */
#define MERCURY 0
#define VENUS 1
#define EARTH 2
#define MARS 3
#define JUPITER 4
#define SATURN 5
#define URANUS 6
#define NEPTUNE 7
#define PLUTO 8
#define MOON 9
#define SUN 10
#define SS_BARY 11
#define EM_BARY 12
#define NUTATIONS 13
#define LIBRATIONS 14
#define TTMTDB 15
#define TCGMTCB 16

/* undefined timescale */
#define CALCEPH_UNDEFINED_TIMESCALE 0

/*-----------------------------------------------------------------*/
/* structures */
/*-----------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*! ephemeris descriptor for any type: binary, SPICE */
struct calcephbin
{
    enum
    {
        CALCEPH_unknown = 0,    /*!< invalid ephemeris file */
        CALCEPH_ebinary,        /*!< ephemeris file of type INPOP/ original JPL DE */
        CALCEPH_espice          /*!< ephemeris file of type SPICE : SPK/PCK */
    } etype;                    /*!< type of the ephemeris */
    union
    {
        struct calcephbin_spice spkernel;   /*!< spice kernel */
        struct calcephbin_inpop binary; /*!< INPOP/JPL format */
    } data;
    struct calceph_locale clocale;  /*!< C locale for the conversion : should be freed on close statement  */
    int relationship_tdb_tt;    /*!< model to use for conversions tt<->tdb */
};

/*----------------------------------------------------------------------------*/
/* structure for parsing dates */
/*----------------------------------------------------------------------------*/
struct calceph_time
{
    enum
    {
        CALCEPH_BAD_DATATYPE = 0,
        CALCEPH_JULIANDATE,
        CALCEPH_CALENDAR
    } datatype;

    union
    {
        struct
        {
            double integerpart;
            double decimalpart;
        } juliandate;
        struct
        {
            int year;
            int month;
            int day;
            int hour;
            int minute;
            double second;
        } calendar;
    } datetime;

    int timescale;
};

/*----------------------------------------------------------------------------*/
/* functions */
/*----------------------------------------------------------------------------*/
t_calcephbin *calceph_open_array2(int n, char **filename);
int calceph_unit_convert_quantities_time(stateType * Planet, int InputUnit, int OutputUnit);
int calceph_unit_convert_quantities_time_array(int n, stateType * Planet, int InputUnit, int OutputUnit);

/*! time parsing */
int calceph_parse_time(struct calceph_locale *pclocale, const char *timestr, struct calceph_time *time);

/*! time conversion */
int calceph_time_get_delta_tt_utc(t_calcephbin * eph, double jd0_tt, double jdfrac_tt, double *delta_tt_utc);

int calceph_time_get_delta_utc_tt(t_calcephbin * eph, double jd0_utc, double jdfrac_utc, double *delta_utc_tt);

int calceph_time_get_delta_tt_tai(t_calcephbin * eph, double *delta_tt_tai);

int calceph_time_get_delta_tai_utc(t_calcephbin * eph, double jd0_utc, double jdfrac_utc, double *delta_tai_utc);

int calceph_time_get_delta_utc_tai(t_calcephbin * eph, double jd0_utc, double jdfrac_utc, double *delta_tai_utc);

int calceph_time_get_delta_tdb_tt(t_calcephbin * eph, double *delta_tdb_tt);

int calceph_time_cal_to_jd_day(int yy, int month, int day, long *jd_integer, double *jd_fraction);

int calceph_time_jd_to_cal_day(double jd0, double jdfrac, int *yy, int *month, int *day, double *fd);

/*----------------------------------------------------------------------------*/
/* get constants */
/*----------------------------------------------------------------------------*/
double calceph_getEMRAT(t_calcephbin * p_pbinfile);
double calceph_getAU(t_calcephbin * p_pbinfile);

/*-----------------------------------------------------------------*/
/*! create/destroy locale independante  */
/*-----------------------------------------------------------------*/
int calceph_locale_init(struct calceph_locale *clocale);
void calceph_locale_clear(struct calceph_locale *clocale);
