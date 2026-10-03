/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeconversiontdbspacecraft.c
  \brief functions that compute spacecraft <-> tdb conversions

  \author  D. De Araujo, M. Gastineau
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

#if HAVE_MATH_H
/* enable M_PI with windows sdk */
#define _USE_MATH_DEFINES
#include <math.h>
#endif
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

/*--------------------------------------------------------------------------*/
/*! Structure to hold Spacecraft Clock (SCLK) kernel information.

    This structure stores all necessary parameters retrieved from a SCLK kernel
    file to perform conversions between SCLK strings and standard time systems.
*/
/*--------------------------------------------------------------------------*/
struct sclk_infos
{
    struct calceph_time id;     /*!< SCLK ID parsed from kernel */
    int type;                   /*!< Data type of the SCLK kernel */
    int time_system;            /*!< Time system used for conversion (1=TDB, 2=TDT) */
    double moduli[10];          /*!< Moduli for the clock fields */
    double offsets[10];         /*!< Offsets for the clock fields */
    int n_fields;               /*!< Number of fields in the SCLK string */
    int output_delim;           /*!< Separator used in SCLK string representation */
    char constant_name[CALCEPH_MAX_CONSTANTNAME];
    int n_partition_start;      /*!< Count of partition start entries */
    double *partition_start;    /*!< Start ticks for each partition */

    int n_partition_end;        /*!< Count of partition end entries */
    double *partition_end;      /*!< End ticks for each partition */

    int n_coeffs;               /*!< Total number of elements in the coefficient array */
    double *coeffs;             /*!< Mapping coefficients: triplets of (SCLK, Time, Rate) */
};

static void clean_sclk_infos(struct sclk_infos *infos);

/*--------------------------------------------------------------------------*/
/*! Retrieve SCLK kernel information for a specific target.

    Reads all necessary constants from the loaded ephemeris kernel to populate
    the sclk_infos structure. This includes clock type, moduli, offsets,
    partitions, and coefficients.

    @return 0 on error, otherwise 1

    @param eph      (in)  ephemeris object
    @param target   (in)  NAIF ID of the spacecraft/target
    @param infos    (out) structure to be filled with SCLK data
*/
/*--------------------------------------------------------------------------*/
static int retrieve_infos_from_sclk(t_calcephbin *eph, int target, struct sclk_infos *infos)
{
    int j;
    t_calcephcharvalue str_id;
    char constant_name[CALCEPH_MAX_CONSTANTNAME];
    double constant_value;

    infos->partition_start = NULL;
    infos->partition_end = NULL;
    infos->coeffs = NULL;

    /* Retrieve SCLK_KERNEL_ID to parse the time ID */
    if (calceph_getconstantss(eph, "SCLK_KERNEL_ID", str_id) == 0)
    {
        fatalerror("SCLK_KERNEL_ID is missing in the kernels file");
        return 0;
    }

    calceph_parse_time(&eph->clocale, str_id, &(infos->id));

    /* Retrieve SCLK_DATA_TYPE (e.g., 1) */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK_DATA_TYPE_%i", target);

    if (calceph_getconstant(eph, constant_name, &constant_value) == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    infos->type = (int) constant_value;

    /* Retrieve SCLK Time System (1=TDB, 2=TDT) */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_TIME_SYSTEM_%i", infos->type, target);

    if (calceph_getconstant(eph, constant_name, &constant_value) == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    infos->time_system = (int) constant_value;

    /* Retrieve number of fields in the clock string */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_N_FIELDS_%i", infos->type, target);

    if (calceph_getconstant(eph, constant_name, &constant_value) == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    infos->n_fields = (int) constant_value;

    /* Retrieve Moduli for the fields */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_MODULI_%i", infos->type, target);

    if (calceph_getconstantvd(eph, constant_name, infos->moduli, infos->n_fields) != infos->n_fields)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    /* Retrieve Offsets for the fields */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_OFFSETS_%i", infos->type, target);

    if (calceph_getconstantvd(eph, constant_name, infos->offsets, infos->n_fields) != infos->n_fields)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    /* Retrieve Output Delimiter */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_OUTPUT_DELIM_%i", infos->type, target);

    if (calceph_getconstant(eph, constant_name, &constant_value) == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    infos->output_delim = (int) constant_value;

    /* Retrieve Partition Start values */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK_PARTITION_START_%i", target);

    infos->n_partition_start = calceph_getconstantvd(eph, constant_name, NULL, 0);
    if (infos->n_partition_start == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }

    infos->partition_start = (double *) malloc(sizeof(double) * infos->n_partition_start);

    if (infos->partition_start == NULL)
    {
        fatalerror("Memory allocation failed for partition_start array (%d elements).\n", infos->n_partition_start);
        return 0;
    }

    calceph_getconstantvd(eph, constant_name, infos->partition_start, infos->n_partition_start);

    /* Retrieve Partition End values */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK_PARTITION_END_%i", target);

    infos->n_partition_end = calceph_getconstantvd(eph, constant_name, NULL, 0);
    if (infos->n_partition_end == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }
    infos->partition_end = (double *) malloc(sizeof(double) * infos->n_partition_end);

    if (infos->partition_end == NULL)
    {
        fatalerror("Memory allocation failed for partition_end array (%d elements).\n", infos->n_partition_end);
        clean_sclk_infos(infos);
        return 0;
    }

    calceph_getconstantvd(eph, constant_name, infos->partition_end, infos->n_partition_end);

    /* Retrieve Partition Coefficients */
    calceph_snprintf(constant_name, CALCEPH_MAX_CONSTANTNAME, "SCLK%02i_COEFFICIENTS_%i", infos->type, target);

    infos->n_coeffs = calceph_getconstantvs(eph, constant_name, NULL, 0);
    if (infos->n_coeffs == 0)
    {
        fatalerror("%s is missing in the kernels file", constant_name);
        return 0;
    }
    infos->coeffs = (double *) malloc(sizeof(double) * infos->n_coeffs);

    if (infos->coeffs == NULL)
    {
        fatalerror("Memory allocation failed for coefficients array (%d elements).\n", infos->n_coeffs);
        clean_sclk_infos(infos);
        return 0;
    }

    int n_coeffs_expected = calceph_getconstantvd(eph, constant_name, infos->coeffs, infos->n_coeffs);

    if (n_coeffs_expected != infos->n_coeffs)
    {
        /* try using string and convert to numbers */
        t_calcephcharvalue *arszcoeffs = (t_calcephcharvalue *) malloc(sizeof(t_calcephcharvalue) * infos->n_coeffs);

        if (arszcoeffs == NULL)
        {
            fatalerror("Memory allocation failed for coefficients array (%d elements).\n", infos->n_coeffs);
            /* Clean up previously allocated memory */
            clean_sclk_infos(infos);
            return 0;
        }
        n_coeffs_expected = calceph_getconstantvs(eph, constant_name, arszcoeffs, infos->n_coeffs);
        if (n_coeffs_expected != infos->n_coeffs)
        {
            fatalerror("Can't load the coefficients of spacecraft clock kernel '%s'.\n", constant_name);
            /* Clean up previously allocated memory */
            free(arszcoeffs);
            clean_sclk_infos(infos);
            return 0;
        }
        for (j = 0; j < n_coeffs_expected; j++)
        {
            if (arszcoeffs[j][0] == '@')
            {
                double jd0, jdfrac;
                int timescale = -1;

                if (infos->time_system == 1)
                    timescale = CALCEPH_TDB;
                else if (infos->time_system == 2)
                    timescale = CALCEPH_TT;

                if (calceph_time_str_to_jd(eph, timescale, &(arszcoeffs[j][1]), &jd0, &jdfrac) == 0)
                {
                    free(arszcoeffs);
                    clean_sclk_infos(infos);
                    return 0;
                }
                infos->coeffs[j] = ((jd0 - 2451545.0) + jdfrac) * 86400.;
            }
            else
            {
                infos->coeffs[j] = calceph_strtod(arszcoeffs[j], NULL, eph->clocale);
            }
        }
        free(arszcoeffs);
    }
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Helper to free memory allocated in sclk_infos */
/*--------------------------------------------------------------------------*/
static void clean_sclk_infos(struct sclk_infos *infos)
{
    if (infos->partition_start)
        free(infos->partition_start);
    if (infos->partition_end)
        free(infos->partition_end);
    if (infos->coeffs)
        free(infos->coeffs);
    infos->partition_start = NULL;
    infos->partition_end = NULL;
    infos->coeffs = NULL;
}

/*--------------------------------------------------------------------------*/
/*! Parse a spacecraft clock string.

    Parses a string formatted as "partition/field1:field2..." to extract the
    partition number and the clock field values (RIM). It handles various
    delimiters (., :, -, ,, space) based on the SCLK kernel definition.

    @return 0 on error, otherwise 1

    @param str       (in)  The input spacecraft clock string
    @param infos     (in)  SCLK info structure containing format definitions
    @param partition (out) Pointer to store the parsed partition number
    @param rim       (out) Array to store the parsed clock field values
*/
/*--------------------------------------------------------------------------*/
static int parse_spacecraft_string(const char *str, struct sclk_infos infos, int *partition, int rim[10])
{
    int pos = 0;

    /* Skip leading whitespaces */
    while (str[pos] == ' ' || str[pos] == '\t')
        pos++;

    if (str[pos] == '\0')
    {
        fatalerror("parse_spacecraft_string: Input spacecraft clock string is empty.\n");
        return 0;
    }

    /* Parse partition number */
    *partition = 0;
    int n_digits_partition = 0;

    while (str[pos] >= '0' && str[pos] <= '9')
    {
        /* Strictly restrict partition to 4 digits */
        if (n_digits_partition >= 4)
        {
            fatalerror("parse_spacecraft_string: Partition number in string '%s' exceeds 4 digits.\n", str);
            return 0;
        }

        *partition *= 10;
        *partition += str[pos] - '0';
        pos++;
        n_digits_partition++;
    }

    /* Skip spaces after partition */
    while (str[pos] == ' ' || str[pos] == '\t')
        pos++;

    /* Check for the mandatory slash separator after partition */
    if (str[pos] != '/')
    {
        fatalerror("parse_spacecraft_string: Missing '/' separator after partition in string '%s'.\n", str);
        return 0;
    }
    pos++;

    /* Determine the expected delimiter character from SCLK info */
    char delim;

    switch (infos.output_delim)
    {
        case 1:
            delim = '.';
            break;
        case 2:
            delim = ':';
            break;
        case 3:
            delim = '-';
            break;
        case 4:
            delim = ',';
            break;
        case 5:
            delim = ' ';
            break;
        default:
            /* Fallback or unknown delimiter type */
            fatalerror("parse_spacecraft_string: Fallback or unknown delimiter type.\n");
            return 1;
    }

    /* Parse the clock fields (RIM) */
    int n_fields = 0;
    int pos_rim = 0;

    while (str[pos] != '\0' && n_fields < infos.n_fields && pos_rim < 10)
    {
        /* Skip spaces before number */
        while (str[pos] == ' ' || str[pos] == '\t')
            pos++;

        rim[pos_rim] = 0;

        /* Parse number for current field */
        int found_digits = 0;

        while (str[pos] >= '0' && str[pos] <= '9')
        {
            found_digits = 1;
            rim[pos_rim] *= 10;
            rim[pos_rim] += str[pos] - '0';

            /* Validate the value against the modulus defined in the kernel */
            if (rim[pos_rim] >= infos.moduli[pos_rim])
            {
                fatalerror("parse_spacecraft_string: Field value %d exceeds modulus %f at index %d in string '%s'.",
                           rim[pos_rim], infos.moduli[pos_rim], pos_rim, str);
                return 0;
            }

            pos++;
        }

        if (!found_digits)
        {
            fatalerror("parse_spacecraft_string: Expected a numeric value for field %d in string '%s', found none.",
                       n_fields, str);
            return 0;
        }

        /* Skip spaces after number */
        while (str[pos] == ' ' || str[pos] == '\t')
            pos++;

        n_fields++;
        pos_rim++;

        /* Check for delimiter if we are not at end of string */
        if (delim != ' ' && str[pos] != '\0')
        {
            if (str[pos] == delim)
            {
                pos++;
            }
            else
            {
                fatalerror("parse_spacecraft_string: Expected delimiter '%c' but found '%c' in string '%s'.", delim,
                           str[pos], str);
                return 0;
            }
        }
    }

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert parsed fields (partition + rim) into a continuous Encoded SCLK value.

    This function computes the total number of "ticks" since the beginning of
    the mission (or clock start) by combining the field values using their
    respective moduli and adding the base tick count of the specified partition.

    @return The total encoded SCLK value (ticks) as a double.
            0.0 on error.

    @param infos     (in) SCLK info structure containing moduli and partitions
    @param partition (in) The partition number (1-based)
    @param rim       (in) Array of integer values for each clock field
*/
/*--------------------------------------------------------------------------*/
static double get_encoded_sclk(struct sclk_infos *infos, int partition, int *rim)
{
    double ticks = 0.0;
    int i;

    /* Validate that the partition index is within the range defined by the kernel */
    if (partition <= 0 || partition > infos->n_partition_start)
    {
        fatalerror("Partition %d is out of range. The SCLK kernel only defines %d partitions.",
                   partition, infos->n_partition_start);
        return 0.0;
    }

    /* Start with the most significant field value */
    ticks = rim[0];

    /* Accumulate ticks from subsequent fields using the moduli */
    /* Formula: Val = Val * Modulus[i] + Field[i] */
    for (i = 1; i < infos->n_fields; i++)
    {
        ticks = ticks * infos->moduli[i] + (double) rim[i];
    }

    /* Add the absolute tick offset for the start of the specified partition.
       Note: Partitions are 1-based in the string, but 0-based in the array. */
    ticks += infos->partition_start[partition - 1];

    return ticks;
}

/*--------------------------------------------------------------------------*/
/*! Convert Spacecraft Clock String to TDB (Julian Date).

    Converts a spacecraft clock string (SCLK) into Barycentric Dynamical Time (TDB).
    The function performs the following steps:
    1. Parses the SCLK string.
    2. Converts fields to continuous ticks.
    3. Interpolates time using the SCLK coefficients (linear segment).
    4. Handles the "Parallel Time System" (usually TDB or TDT).
    5. Converts the result to TDB if the parallel system was TDT.

    @return 0 on error, otherwise non-zero value

    @param eph          (in)  ephemeris object
    @param target       (in)  NAIF ID of the spacecraft
    @param str          (in)  Spacecraft clock string (e.g., "1/1234:56")
    @param jd0_tdb      (out) Integer part of Julian Date (TDB)
    @param jdfrac_tdb   (out) Fractional part of Julian Date (TDB)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_str_spacecraft_clock_to_jd_tdb(t_calcephbin *eph, int target, const char *str, double *jd0_tdb,
                                                double *jdfrac_tdb)
{
    struct sclk_infos infos;
    int partition;
    int rim[10];
    int status;
    double encoded_sclk;
    double ref_sclk, ref_par, rate;
    double msf;
    int j, k, rec_idx, n_records;

    /* Intermediate variables for the parallel time system (TDB or TDT) */
    double par_jd0, par_jdfrac;

    /* Initialize array */
    for (j = 0; j < 10; j++)
        rim[j] = 0;

    /* Check input validity */
    if (eph == NULL)
    {
        fatalerror("calceph_time_str_spacecraft_clock_to_jd_tdb: Ephemeris descriptor is NULL.");
        return 0;
    }

    /* 1. Retrieve SCLK kernel information */
    if (retrieve_infos_from_sclk(eph, target, &infos) == 0)
    {
        /* Error is reported inside retrieve_infos_from_sclk */
        clean_sclk_infos(&infos);
        return 0;
    }

    /* 2. Parse the input string into partition and fields */
    if (parse_spacecraft_string(str, infos, &partition, rim) == 0)
    {
        /* Error is reported inside parse_spacecraft_string */
        clean_sclk_infos(&infos);
        return 0;
    }

    /* 3. Convert fields to total ticks */
    encoded_sclk = get_encoded_sclk(&infos, partition, rim);

    /* 4. Find the applicable coefficient record */
    /* The coefficients array is a list of triplets: [SCLK_REF, TIME_REF, RATE] */
    n_records = infos.n_coeffs / 3;
    rec_idx = 0;

    for (j = 0; j < n_records - 1; j++)
    {
        /* Check if the encoded time falls before the start of the next segment */
        if (encoded_sclk < infos.coeffs[3 * (j + 1)])
        {
            rec_idx = j;
            break;
        }
        /* Fallback to the last record if we reach the end */
        rec_idx = j + 1;
    }

    /* Extract coefficients for the found interval */
    ref_sclk = infos.coeffs[3 * rec_idx];
    ref_par = infos.coeffs[3 * rec_idx + 1];    /* Reference time in Parallel System (sec past J2000) */
    rate = infos.coeffs[3 * rec_idx + 2];   /* Rate (Time / Tick) */

    /* 5. Compute the MSF factor (Ticks per principal unit) */
    /* This robustly handles clocks with N fields by multiplying all fine moduli */
    msf = 1.0;
    for (k = 1; k < infos.n_fields; k++)
    {
        msf *= infos.moduli[k];
    }

    /* 6. Calculate the time delta in the Parallel System */
    /* We perform the calculation carefully to preserve precision */
    double delta_ticks = encoded_sclk - ref_sclk;

    /* Separate whole units (seconds-like) and remaining ticks */
    double delta_whole_unit = floor(delta_ticks / msf);
    double delta_rem_ticks = delta_ticks - (delta_whole_unit * msf);

    /* Apply rate */
    delta_whole_unit *= rate;
    delta_rem_ticks *= rate;
    double delta_frac_unit = delta_rem_ticks / msf;

    /* 7. Reconstruct the absolute date (Split Double Algorithm) */
    /* Reference time (ref_par) is in seconds past J2000 */
    double ref_days_int = floor(ref_par / 86400.0);
    double ref_sec_rem = ref_par - (ref_days_int * 86400.0);

    /* Delta time is in seconds */
    double delta_days_int = floor(delta_whole_unit / 86400.0);
    double delta_sec_rem = delta_whole_unit - (delta_days_int * 86400.0);

    /* Sum integer parts (J2000 offset + Reference + Delta) */
    par_jd0 = 2451545.0 + ref_days_int + delta_days_int;

    /* Sum fractional parts (seconds) */
    double total_rem_seconds = ref_sec_rem + delta_sec_rem + delta_frac_unit;

    par_jdfrac = total_rem_seconds / 86400.0;

    /* Normalize the result (handle fraction overflow) */
    double final_int;
    double final_frac = modf(par_jdfrac, &final_int);

    par_jd0 += final_int;
    par_jdfrac = final_frac;

    /* 8. Final Conversion: Parallel System -> TDB */
    /* If the SCLK is defined against TT (Time System 2), convert to TDB. */
    if (infos.time_system == 2)
    {
        /* The calculated time is TT, convert to TDB */
        /* Note: 'eph' must contain the necessary LSK/Time kernels */
        status = calceph_time_jd_tt_to_jd_tdb(eph, par_jd0, par_jdfrac, jd0_tdb, jdfrac_tdb);

        if (status == 0)
        {
            clean_sclk_infos(&infos);
            return 0;
        }
    }
    else
    {
        /* The SCLK is already defined against TDB (Time System 1) */
        *jd0_tdb = par_jd0;
        *jdfrac_tdb = par_jdfrac;
    }

    clean_sclk_infos(&infos);
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert TDB (Julian Date) to Spacecraft Clock String.

    Computes the spacecraft clock string corresponding to a given Barycentric
    Dynamical Time (TDB). The function performs the reverse operation of the
    SCLK parsing:
    1. Converts TDB to the parallel time system (e.g., TT) if required.
    2. Finds the appropriate rate coefficient record.
    3. Computes the elapsed time and converts it to clock ticks.
    4. Identifies the correct partition.
    5. Formats the ticks into fields (RIM:MOD) and produces the output string.

    @return 0 on error, otherwise non-zero value

    @param eph          (in)  ephemeris object
    @param target       (in)  NAIF ID of the spacecraft
    @param jd0_tdb      (in)  Integer part of Julian Date (TDB)
    @param jdfrac_tdb   (in)  Fractional part of Julian Date (TDB)
    @param str          (out) Output string buffer (must be large enough, e.g., 64 chars)
*/
/*--------------------------------------------------------------------------*/
int calceph_time_jd_tdb_to_str_spacecraft_clock(t_calcephbin *eph, int target, double jd0_tdb, double jdfrac_tdb,
                                                t_calcephcharvalue str)
{
    struct sclk_infos infos;
    int p, i, j, k, rec_idx, n_records;
    double encoded_sclk;
    double ref_sclk, ref_par, rate;
    double ticks_in_partition, msf;

    /* Working variables for the parallel time system (TDB or TDT) */
    double work_jd0, work_jdfrac;
    double work_seconds_from_j2000;
    int status;

    /* Variables for high-precision time difference calculation */
    double ref_days, ref_sec_rem;
    double in_days, in_sec;
    double delta_days, delta_sec, total_delta_sec;

    int fields[10];
    int widths[10];
    char delim_char;
    char buffer[64];
    char fmt[16];

    /* Check input validity */
    if (eph == NULL)
    {
        fatalerror("Ephemeris descriptor is NULL.");
        return 0;
    }

    /* Retrieve SCLK kernel information */
    if (retrieve_infos_from_sclk(eph, target, &infos) == 0)
    {
        clean_sclk_infos(&infos);
        return 0;
    }

    /* 1. CONVERT TDB -> PARALLEL TIME SYSTEM */
    if (infos.time_system == 2)
    {
        /* The kernel expects TDT (TT), but input is TDB. Convert it. */
        /* 'eph' must contain the necessary LSK/Time kernels. */
        status = calceph_time_jd_tdb_to_jd_tt(eph, jd0_tdb, jdfrac_tdb, &work_jd0, &work_jdfrac);

        if (status == 0)
        {
            clean_sclk_infos(&infos);
            return 0;
        }
    }
    else
    {
        /* The kernel is already defined against TDB */
        work_jd0 = jd0_tdb;
        work_jdfrac = jdfrac_tdb;
    }

    /* Calculate seconds from J2000 in the parallel system (used for lookup) */
    work_seconds_from_j2000 = (work_jd0 - 2451545.0) * 86400.0 + work_jdfrac * 86400.0;

    /* 2. FIND COEFFICIENT RECORD */
    /* Search for the interval where Time_Ref <= Current_Time */
    n_records = infos.n_coeffs / 3;
    rec_idx = 0;

    for (j = 0; j < n_records - 1; j++)
    {
        /* Compare against the time value of the NEXT record (index 1 of triplet) */
        if (work_seconds_from_j2000 < infos.coeffs[3 * (j + 1) + 1])
        {
            rec_idx = j;
            break;
        }
        rec_idx = j + 1;
    }

    ref_sclk = infos.coeffs[3 * rec_idx];
    ref_par = infos.coeffs[3 * rec_idx + 1];
    rate = infos.coeffs[3 * rec_idx + 2];

    /* 3. CALCULATE DELTA TIME (High Precision) */
    /* To minimize precision loss, we handle days and seconds separately */

    /* Decompose Reference Time */
    ref_days = floor(ref_par / 86400.0);
    ref_sec_rem = ref_par - (ref_days * 86400.0);

    /* Decompose Input Time (relative to J2000) */
    in_days = work_jd0 - 2451545.0;
    in_sec = work_jdfrac * 86400.0;

    /* Calculate differences */
    delta_days = in_days - ref_days;
    delta_sec = in_sec - ref_sec_rem;

    /* Recombine to get total delta in seconds */
    total_delta_sec = delta_days * 86400.0 + delta_sec;

    /* 4. COMPUTE MSF (Total ticks per principal unit) */
    msf = 1.0;
    for (k = 1; k < infos.n_fields; k++)
    {
        msf *= infos.moduli[k];
    }

    /* 5. CALCULATE ENCODED SCLK (Total Ticks) */
    /* Formula: SCLK = RefSclk + (DeltaTime / Rate) * TicksPerUnit */
    encoded_sclk = ref_sclk + (total_delta_sec / rate) * msf;

    /* Round to the nearest integer tick to avoid off-by-one errors due to float precision */
    encoded_sclk = floor(encoded_sclk + 0.5);

    /* 6. IDENTIFY PARTITION */
    int partition = 0;

    for (p = 0; p < infos.n_partition_start; p++)
    {
        /* Check if ticks fall within the bounds of partition p */
        if (encoded_sclk >= infos.partition_start[p] && encoded_sclk <= infos.partition_end[p])
        {
            partition = p + 1;  /* Partitions are 1-based in output string */
            break;
        }
    }

    if (partition == 0)
    {
        fatalerror("Calculated SCLK value (%.2f) does not fall into any known partition boundaries.", encoded_sclk);
        clean_sclk_infos(&infos);
        return 0;
    }

    /* 7. ENCODE FIELDS (RIM:MOD) */
    /* Get ticks relative to the start of the partition */
    ticks_in_partition = encoded_sclk - infos.partition_start[partition - 1];

    /* Decompose total ticks into fields from right to left (Fine -> Coarse) */
    for (i = infos.n_fields - 1; i > 0; i--)
    {
        fields[i] = (int) fmod(ticks_in_partition, infos.moduli[i]);
        ticks_in_partition = floor(ticks_in_partition / infos.moduli[i]);
    }
    fields[0] = (int) ticks_in_partition;

    /* 8. FORMAT OUTPUT STRING */

    /* Calculate required width for zero-padding based on modulus */
    for (i = 0; i < infos.n_fields; i++)
    {
        double max_val = infos.moduli[i] - 1.0;
        int w = 1;

        while (max_val >= 10.0)
        {
            max_val /= 10.0;
            w++;
        }
        widths[i] = w;
    }

    /* Determine delimiter character */
    switch (infos.output_delim)
    {
        case 1:
            delim_char = '.';
            break;
        case 2:
            delim_char = ':';
            break;
        case 3:
            delim_char = '-';
            break;
        case 4:
            delim_char = ',';
            break;
        case 5:
            delim_char = ' ';
            break;
        default:
            delim_char = ':';
            break;
    }

    /* Construct the string: "Partition/Field0:Field1..." */
    calceph_snprintf(str, CALCEPH_MAX_CONSTANTNAME, "%d/", partition);

    for (i = 0; i < infos.n_fields; i++)
    {
        /* Create dynamic format string for padding (e.g., "%03d") */
        calceph_snprintf(fmt, 16, "%%0%dd", widths[i]);
        calceph_snprintf(buffer, 64, fmt, fields[i]);
        strcat(str, buffer);

        /* Add delimiter if not the last field */
        if (i < infos.n_fields - 1)
        {
            size_t len = strlen(str);

            str[len] = delim_char;
            str[len + 1] = '\0';
        }
    }

    clean_sclk_infos(&infos);
    return 1;
}
