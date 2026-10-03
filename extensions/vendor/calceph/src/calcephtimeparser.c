/*-----------------------------------------------------------------*/
/*!
  \file calcephtimeparser.c
  \brief internal functions for parsing datetime strings

  \author  D. De Araujo, M. Gastineau
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

#include "calcephconfig.h"

#if HAVE_MATH_H
/* enable M_PI with windows sdk */
#define _USE_MATH_DEFINES
#include <math.h>
#endif
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
#include "calcephtimelexer.h"

/*--------------------------------------------------------------------------*/
/*! struct time_specifiers
 *  Stores additional time-related qualifiers detected during parsing.
 *
 *  - ampm : 1 = AM, 2 = PM, 0 = unspecified
 *  - era  : 1 = BC, 2 = AD, 0 = unspecified
 *  - timezone : "UTC+2", "PST", etc.
 *  - timescale  : UTC, TDB, etc.
 */
/*--------------------------------------------------------------------------*/
struct time_specifiers
{
    int ampm;
    int era;
    t_calcephcharvalue timezone;
    int timescale;
};

/*--------------------------------------------------------------------------*/
/*! Apply initial transformations to lexer tokens.
 *
 *  This function normalizes certain token patterns right after tokenization:
 *    - Converts UTC offset forms (e.g., "UTC+i:i", "UTC-i") into timezone tokens.
 *    - Removes trailing dots after weekdays and months.
 *    - Merges integer sequences into numeric tokens.
 *    - Removes whitespace tokens.
 *
 *  @param  lexer  (in/out)  lexer to normalize
 */
/*--------------------------------------------------------------------------*/
static void initial_token_processing(struct lexer *lexer)
{
    /* Normalize UTC offset forms */
    calceph_lexer_replace_tokens_from_pattern(lexer, "Oi:i", "Z<<<", 1);    /* UTC+i:i */
    calceph_lexer_replace_tokens_from_pattern(lexer, "Oi", "Z<", 1);    /* UTC+i */
    calceph_lexer_replace_tokens_from_pattern(lexer, "oi:i", "Z<<<", 1);    /* UTC-i:i */
    calceph_lexer_replace_tokens_from_pattern(lexer, "oi", "Z<", 1);    /* UTC-i */

    /* Remove trailing dots after weekdays and months */
    calceph_lexer_replace_tokens_from_pattern(lexer, "w.", "w<", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "m.", "m<", 1);

    /* Merge integer tokens into NUMBER tokens */
    calceph_lexer_replace_tokens_from_pattern(lexer, "i.i", "n<<", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i.", "n<", 1);

    /* Remove whitespace tokens */
    calceph_lexer_replace_tokens_from_pattern(lexer, "b", "*", 1);
}

/*--------------------------------------------------------------------------*/
/*! Parse a Julian Date (JD) expression from a lexer.
 *
 *  This function scans the tokenized input to identify and extract a Julian
 *  Date value. The numeric token is converted into 2 double precision values
 *  and stored in the output structure. If present, the time system token
 *  (UTC, TDB, TCB, TAI, TT) is also processed.
 *
 *  Parsing steps:
 *    1. Normalize JD and time system related tokens (remove brackets).
 *    2. Convert integer tokens to NUMBER if needed.
 *    3. Handle time system token (TDB, TCB, UTC, TAI, TT).
 *    4. Split the numeric string into integer and decimal parts.
 *
 *  @return 1 if a valid Julian Date was successfully parsed, 0 if not valid, and -1 on error.
 *
 *  @param lexer   (in/out) tokenized representation of the input string
 *  @param timestr (in)     original input time string
 *  @param time    (out)    structure receiving the parsed Julian Date
 */
/*--------------------------------------------------------------------------*/
static int parse_jd(struct lexer *lexer, const char *timestr, struct calceph_time *time)
{
    struct token *tok;

    struct token *tmptok;

    t_calcephcharvalue tokvalue;

    /* Check presence of 'j' token (Julian Date marker) */
    if (calceph_lexer_replace_tokens_from_pattern(lexer, "j", "j", 1) == 0)
        return 0;

    /* Step 1: Normalize [s] and [j] tokens */
    calceph_lexer_replace_tokens_from_pattern(lexer, "[s]", "*s*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "[j]", "*j*", 1);

    /* Step 2: If no NUMBER token is found, convert the leftmost INTEGER to NUMBER */
    struct token *tokint = NULL;

    int hastokn = calceph_lexer_replace_tokens_from_pattern(lexer, "n", "n", 1);

    for (tok = lexer->head; tok != NULL && hastokn == 0; tok = tok->next)
    {
        if (tok->type == TOKEN_INTEGER && tokint == NULL)
        {
            tokint = tok;
        }
    }

    if (tokint != NULL && hastokn == 0)
    {
        tokint->type = TOKEN_NUMBER;
    }

    /* Step 3: Normalize negative numbers (-n -> n) */
    calceph_lexer_replace_tokens_from_pattern(lexer, "-n", "n<", 1);

    /* Step 4: Remove the 'j' (JD marker) token */
    calceph_lexer_replace_tokens_from_pattern(lexer, "j", "*", 1);

    /* Step 5: Process time system token (TDB, TCB, UTC, TAI, TT) if present */
    time->timescale = CALCEPH_UNDEFINED_TIMESCALE;

    tok = lexer->head;

    int timescale_found = 0;

    while (tok != NULL && timescale_found == 0)
    {
        if (tok->type == TOKEN_TIMESCALE)
        {
            timescale_found = 1;
            calceph_token_read(tok, timestr, tokvalue);

            /* Identify time system */
            if (strcmp(tokvalue, "TDB") == 0)
            {
                time->timescale = CALCEPH_TDB;
            }
            else if (strcmp(tokvalue, "TCB") == 0)
            {
                time->timescale = CALCEPH_TCB;
            }
            else if (strcmp(tokvalue, "UTC") == 0)
            {
                time->timescale = CALCEPH_UTC;
            }
            else if (strcmp(tokvalue, "TAI") == 0)
            {
                time->timescale = CALCEPH_TAI;
            }
            else if (strcmp(tokvalue, "TT") == 0)
            {
                time->timescale = CALCEPH_TT;
            }
            else
            {
                /* Invalid time system */
                time->timescale = CALCEPH_UNDEFINED_TIMESCALE;
                time->datatype = CALCEPH_BAD_DATATYPE;
                return -1;
            }

            /* Remove the timescale token from lexer */
            tmptok = tok->next;
            calceph_lexer_remove_token(lexer, tok);
            tok = tmptok;
        }
        else
        {
            tok = tok->next;
        }
    }

    /* Default to UTC if no time system was specified */
    if (time->timescale == CALCEPH_UNDEFINED_TIMESCALE)
    {
        time->timescale = CALCEPH_UTC;
    }

    /* Check: must have exactly one NUMBER token */
    if (lexer->head == NULL || lexer->ntokens != 1 || lexer->head->type != TOKEN_NUMBER)
        return -1;

    tok = lexer->head;
    calceph_token_read(tok, timestr, tokvalue);

    /* Final step: Extract and split the numeric token (integer / fractional parts) */
    char *p_tokvalue = tokvalue;
    double sign = 1.0;

    /* Handle negative sign */
    if (tokvalue[0] == '-')
    {
        sign = -1.0;
        p_tokvalue++;
    }

    const char *decimal_point = strchr(p_tokvalue, '.');
    char int_str[64];
    char frac_str[64];

    if (decimal_point != NULL)
    {
        /* Decimal point was found */
        size_t int_len = decimal_point - p_tokvalue;

        /* Copy integer part */
        if (int_len > 63)
            int_len = 63;
        strncpy(int_str, p_tokvalue, int_len);
        int_str[int_len] = '\0';

        /* Copy fractional part, starting with "0." */
        frac_str[0] = '0';
        strncpy(frac_str + 1, decimal_point, 62);   /* Copies the "." and rest */
        frac_str[63] = '\0';

        /* Convert separately and apply sign */
        time->datetime.juliandate.integerpart = calceph_strtod(int_str, NULL, *lexer->ephlocale) * sign;
        time->datetime.juliandate.decimalpart = calceph_strtod(frac_str, NULL, *lexer->ephlocale) * sign;
    }
    else
    {
        time->datetime.juliandate.integerpart = calceph_strtod(p_tokvalue, NULL, *lexer->ephlocale) * sign;
        time->datetime.juliandate.decimalpart = 0.0;
    }

    time->datatype = CALCEPH_JULIANDATE;
    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Check and parse an ISO formatted date/time expression
 *
 *  @return 1 if the input matches a valid ISO expression, 0 otherwise
 *
 *  @param lexer   (in/out) tokenized representation of the input string
 *  @param timestr (in)     original input string
 *  @param Y       (out)    year
 *  @param m       (out)    month
 *  @param D       (out)    day of month
 *  @param y       (out)    day of year
 *  @param H       (out)    hour
 *  @param M       (out)    minute
 *  @param S       (out)    second (can be fractional)
 */
/*--------------------------------------------------------------------------*/
static int check_match_iso(struct lexer *lexer, const char *timestr,
                           int *Y, int *m, double *D, double *y, double *H, double *M, double *S)
{
    /* Verify arguments */
    if (lexer == NULL || timestr == NULL || Y == NULL || m == NULL ||
        D == NULL || y == NULL || H == NULL || M == NULL || S == NULL)
        return 0;

    struct token *tok;
    t_calcephcharvalue tokvalue;
    int before_t_found = 0;
    int after_t_found = 0;
    int j;

    /* Patterns before 'T' — date components */
    char *before_t_tokens[4] = { "Y-i-it", "i-i-it", "Y-it", "i-it" };
    char *before_t_transform[4] = { "Y*m*Dt", "Y*m*Dt", "Y*yt", "Y*yt" };

    /* Patterns after 'T' — time components (hour/min/sec, optional Z or TZ offset) */
    char *after_t_tokens[12] = { "ti:i:iz", "ti:i:nz", "ti:iz", "ti:nz", "tiz", "tnz",
        "ti:i:i", "ti:i:n", "ti:i", "ti:n", "ti", "tn"
    };
    char *after_t_transform[12] = { "*H*M*S*", "*H*M*S*", "*H*M*", "*H*M*", "*H*", "*H*",
        "*H*M*S", "*H*M*S", "*H*M", "*H*M", "*H", "*H"
    };

    /* Step 1: Ensure presence of 'T' separator */
    if (calceph_lexer_replace_tokens_from_pattern(lexer, "t", "t", 1) != 1)
    {
        return 0;
    }

    /* Step 2: Try matching date pattern before 'T' */
    for (j = 0; j < 4 && before_t_found == 0; j++)
        before_t_found = calceph_lexer_replace_tokens_from_pattern(lexer, before_t_tokens[j], before_t_transform[j], 1);

    if (before_t_found == 0)
    {
        return 0;               /* no valid date part found */
    }

    /* Step 3: Try matching time pattern after 'T' */
    for (j = 0; j < 12 && after_t_found == 0; j++)
        after_t_found = calceph_lexer_replace_tokens_from_pattern(lexer, after_t_tokens[j], after_t_transform[j], 1);

    /* Step 4: Extract matched components into numeric outputs */
    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        switch (tok->type)
        {
            case TOKEN_YEAR:
                calceph_token_read(tok, timestr, tokvalue);
                *Y = atoi(tokvalue);
                break;
            case TOKEN_MONTH:
                calceph_token_read(tok, timestr, tokvalue);
                *m = atoi(tokvalue);
                break;
            case TOKEN_DAY_OF_MONTH:
                calceph_token_read(tok, timestr, tokvalue);
                *D = calceph_strtod(tokvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_DAY_OF_YEAR:
                calceph_token_read(tok, timestr, tokvalue);
                *y = calceph_strtod(tokvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_HOUR:
                calceph_token_read(tok, timestr, tokvalue);
                *H = calceph_strtod(tokvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_MINUTE:
                calceph_token_read(tok, timestr, tokvalue);
                *M = calceph_strtod(tokvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_SECOND:
                calceph_token_read(tok, timestr, tokvalue);
                *S = calceph_strtod(tokvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_Z:
                /* 'Z' allowed only as final token */
                if (tok->next != NULL)
                    return 0;
                break;
            case TOKEN_T_SEPARATOR:
                if (after_t_found == 0 && tok->next != NULL && tok->next->type != TOKEN_Z)
                    return 0;
                break;
            default:
                /* Unexpected token: not a valid ISO expression */
                return 0;
        }
    }

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Convert a day-of-year value into (month, day) in the Gregorian calendar.
 *
 *  @param Y  (in)      year (Gregorian)
 *  @param y  (in)      day of year (can include a fractional part for partial days)
 *  @param m  (out)     month number (1–12, or -1 if invalid)
 *  @param D  (out)     day of month (1–31, or -1 if invalid, may include fractional part)
 */
/*--------------------------------------------------------------------------*/
static void compute_day_month_from_dayofyear(int Y, double y, int *m, double *D)
{
    /* Base number of days per month (non-leap year) */
    static const int days_in_month[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

    int dim[12];

    int j;

    for (j = 0; j < 12; j++)
        dim[j] = days_in_month[j];

    /* Adjust for leap year */
    int leap = ((Y % 4 == 0 && Y % 100 != 0) || (Y % 400 == 0));

    if (leap)
        dim[1] = 29;

    /* Validate day-of-year range */
    int maxdays = leap ? 366 : 365;

    if (y < 1 || y > maxdays)
    {
        *m = -1;
        *D = -1;
        return;
    }

    /* Convert to integer day count (include fractional offset if needed) */
    int daycount = y;

    if (floor(y) != y)
        daycount++;

    /* Find month by subtracting month lengths until remainder fits */
    int month = 0;

    while (month < 12 && daycount > dim[month])
    {
        daycount -= dim[month];
        month++;
    }

    /* Compute final month and fractional day */
    *m = month + 1;
    *D = (double) daycount + (y - floor(y));
}

/*--------------------------------------------------------------------------*/
/*! Compute calendar date and time fields from decimal inputs.
 *
 *  Processing logic:
 *    - If dayofyear != 0, convert it to (month, day) using
 *      compute_day_month_from_dayofyear().
 *    - If day != 0, decompose its fractional part into hours/minutes/seconds.
 *    - Otherwise, fall back to hour/minute/second decomposition.
 *    - Handles fractional parts consistently (e.g. 1.5 days -> 1 day + 12h).
 *
 *  @return 0 on success, 1 if no valid date was provided
 *
 *  @param time         (in/out)    pointer to calceph_time structure to fill
 *  @param dayofyear    (in)        day of year (0 if unused)
 *  @param day          (in)        day of month (may include fraction, 0 if unused)
 *  @param hour         (in)        hour (may include fraction)
 *  @param minute       (in)        minute (may include fraction)
 *  @param second       (in)        second
 */
/*--------------------------------------------------------------------------*/
static int compute_decimal_time(struct calceph_time *time, double dayofyear, double day,
                                double hour, double minute, double second)
{
    double rest;

    /* Convert day-of-year to month/day if applicable */
    if (dayofyear != 0)
    {
        compute_day_month_from_dayofyear(time->datetime.calendar.year, dayofyear, &time->datetime.calendar.month, &day);
    }

    /* If a day is provided, decompose fractional parts */
    if (day != 0)
    {
        /* Extract integer part of the day */
        time->datetime.calendar.day = (int) (floor(day));

        /* If day has a fractional part -> convert remainder to hours/minutes/seconds */
        if (floor(day) != day)
        {
            rest = (day - floor(day)) * 24.;
            time->datetime.calendar.hour = floor(rest);
            rest = (rest - floor(rest)) * 60.;
            time->datetime.calendar.minute = floor(rest);
            rest = (rest - floor(rest)) * 60.;
            time->datetime.calendar.second = rest;
        }
        else
        {
            /* No fractional day -> use provided hour/minute/second */
            time->datetime.calendar.hour = (int) (floor(hour));

            /* Fractional hour -> expand into minutes/seconds */
            if (floor(hour) != hour)
            {
                rest = (hour - floor(hour)) * 60.;
                time->datetime.calendar.minute = floor(rest);
                rest = (rest - floor(rest)) * 60.;
                time->datetime.calendar.second = rest;
            }
            else
            {
                /* Fractional minute -> expand into seconds */
                time->datetime.calendar.minute = (int) (floor(minute));
                if (floor(minute) != minute)
                {
                    time->datetime.calendar.second = (minute - floor(minute)) * 60.;
                }
                else
                {
                    time->datetime.calendar.second = second;
                }
            }
        }
    }
    else
    {
        return 1;
    }

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Parse an ISO calendar date-time expression.
 *
 *  @return 1 if parsing succeeded (valid ISO format), 0 otherwise
 *
 *  @param lexer    (in/out)    pre-tokenized date string
 *  @param timestr  (in)        original date string
 *  @param time     (out)       parsed result (calendar date in UTC)
 */
/*--------------------------------------------------------------------------*/
static int parse_iso(struct lexer *lexer, const char *timestr, struct calceph_time *time)
{
    struct token *tok;

    int isiso = 0;

    int Y = 0, m = 0;
    double y = 0., H = 0., M = 0., S = 0., D = 0.;

    /* Check for unsupported ISO extensions (s, Z, e, N) */
    /* If any present, mark as invalid and return        */
    isiso += calceph_lexer_replace_tokens_from_pattern(lexer, "s", "s", 1);
    isiso += calceph_lexer_replace_tokens_from_pattern(lexer, "Z", "Z", 1);
    isiso += calceph_lexer_replace_tokens_from_pattern(lexer, "e", "e", 1);
    isiso += calceph_lexer_replace_tokens_from_pattern(lexer, "N", "N", 1);

    if (isiso != 0)
    {
        time->datatype = CALCEPH_BAD_DATATYPE;
        return 0;
    }

    /* Convert integers of 4+ digits into year tokens */
    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        if (tok->type == TOKEN_INTEGER && (tok->end - tok->start) >= 3)
        {
            tok->type = TOKEN_YEAR;
        }
    }

    /* Attempt to match ISO format and extract components */
    isiso = check_match_iso(lexer, timestr, &Y, &m, &D, &y, &H, &M, &S);

    if (isiso == 1)
    {
        time->datatype = CALCEPH_CALENDAR;
        time->datetime.calendar.year = Y;
        time->datetime.calendar.month = m;
        compute_decimal_time(time, y, D, H, M, S);
        time->timescale = CALCEPH_UTC;
    }

    return isiso;
}

/*--------------------------------------------------------------------------*/
/*! Convert a WEEKDAY token into its numeric representation.
 *
 *  @return weekday number [1..7], or 0 if no match
 *
 *  @param token    (in)    pointer to the token to analyze
 *  @param timestr  (in)    original input string (used to extract substring)
 */
/*--------------------------------------------------------------------------*/
static int get_weekday_number(struct token *token, const char *timestr)
{
    char *weekdays[7] = { "MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN" };

    int j;
    int res = 0;
    t_calcephcharvalue tokenvalue;

    calceph_token_read(token, timestr, tokenvalue);

    for (j = 0; j < 7 && res == 0; j++)
    {
        if (strncmp(weekdays[j], tokenvalue, 3) == 0)
        {
            res = j + 1;        /* Monday = 1 ... Sunday = 7 */
        }
    }

    return res;
}

/*--------------------------------------------------------------------------*/
/*! Convert a MONTH token into its numeric representation.
 *
 * @return month number [1..12], or 0 if no match
 *
 * @param token     (in)    pointer to the token to analyze
 * @param timestr   (in)    original input string (used to extract substring)
 */
/*--------------------------------------------------------------------------*/
static int get_month_number(struct token *token, const char *timestr)
{
    char *months[12] = { "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
        "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
    };

    int j;
    int res = 0;
    t_calcephcharvalue tokenvalue;

    calceph_token_read(token, timestr, tokenvalue);

    if (atoi(tokenvalue) > 0)
        return atoi(tokenvalue);

    calceph_strtoupper(tokenvalue);

    for (j = 0; j < 12 && res == 0; j++)
    {
        if (strncmp(months[j], tokenvalue, 3) == 0)
        {
            res = j + 1;        /* January = 1 ... December = 12 */
        }
    }

    return res;
}

/*--------------------------------------------------------------------------*/
/*! Try to identify a valid transformation pattern in the lexer token list.
 *
 * @return 1 if a valid transformation is found, 0 otherwise
 *
 * @param lexer             (in/out)    the lexer with tokenized date/time
 * @param transformation    (out)       output string storing the detected pattern
 */
/*--------------------------------------------------------------------------*/
static struct lexer *other_calendar_find_transformation(struct lexer *lexer, t_calcephcharvalue transformation)
{
    /* canonical transformation patterns for dates and times */
    char *transformarray[31] = {
        "DmHMSY",
        "DmHMY",
        "DmY",
        "DmYH",
        "DmYHM",
        "DmYHMS",
        "HMDmY",
        "HMSDmY",
        "HMSmDY",
        "HMmDY",
        "YDm",
        "YDmH",
        "YDmHM",
        "YDmHMS",
        "YmD",
        "YmDH",
        "YmDHM",
        "YmDHMS",
        "Yy",
        "YyH",
        "YyHM",
        "YyHMS",
        "mDHMSY",
        "mDHMY",
        "mDY",
        "mDYH",
        "mDYHM",
        "mDYHMS",
        "yY",
        "yYHM",
        "yYHMS"
    };

    /* mapping of token patterns to normalized date parts */
    const char *datetable[34][2] = {
        {"Y-i-it", "Y*m*D*"},
        {"Y-i/", "Y*y*"},
        {"i-i/", "Y*y*"},
        {"Y-it", "Y*y*"},
        {"Y-id", "Y*y*"},
        {"Yid", "Yy*"},
        {"Yii", "YmD"},
        {"Yim", "YDm"},
        {"Yin", "YmD"},
        {"Ymi", "YmD"},
        {"Ymn", "YmD"},
        {"Ynm", "YDm"},
        {"imY", "DmY"},
        {"iim", "YDm"},
        {"i-Y/", "y*Y*"},
        {"i-Yd", "y*Y*"},
        {"iYd", "yY*"},
        {"iid", "Yy*"},
        {"i-i-Y", "m*D*Y"},
        {"i-i-it", "Y*m*D*"},
        {"i/i/Y", "m*D*Y"},
        {"i/i/i", "m*D*Y*"},
        {"miY", "mDY"},
        {"iiY", "mDY"},
        {"imi:", "Dmi:"},
        {"imi", "YmD"},
        {"imn", "YmD"},
        {"inY", "mDY"},
        {"inm", "YDm"},
        {"mii:", "mDi:"},
        {"mii", "mDY"},
        {"mny", "mDY"},
        {"mni", "mDY"},
        {"nmy", "DmY"}
    };

    int j;
    t_calcephcharvalue strlex;
    int pattern_found = 0;
    int date_pattern_found = 0;

    /* Copy the lexer to avoid modifying the original */
    struct lexer *cplexer = calceph_lexer_copy(lexer);

    calceph_lexer_to_string(lexer, strlex);

    /* Replace date part using datetable mappings */
    /* Stop at the first successful replacement   */
    for (j = 0; j < 33 && date_pattern_found == 0; j++)
    {
        if (calceph_lexer_replace_tokens_from_pattern(cplexer, datetable[j][0], datetable[j][1], 1) == 1)
        {
            date_pattern_found = 1;
        }
    }

    /* Replace time components with standardized tokens */
    calceph_lexer_replace_tokens_from_pattern(cplexer, "i:i:n", "H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "i:i:i", "H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "iin", "HMS", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "iii", "HMS", 1);

    calceph_lexer_replace_tokens_from_pattern(cplexer, "i:n", "H*M", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "i:i", "H*M", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "in", "HM", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "ii", "HM", 1);

    calceph_lexer_replace_tokens_from_pattern(cplexer, "n", "H", 1);
    calceph_lexer_replace_tokens_from_pattern(cplexer, "i", "H", 1);

    calceph_lexer_to_string(cplexer, strlex);

    /* Compare transformed string against known patterns */
    for (j = 0; j < 31 && pattern_found == 0; j++)
    {
        if (strcmp(transformarray[j], strlex) == 0)
        {
            strcpy(transformation, transformarray[j]);
            pattern_found = 1;
        }
    }

    /* If no pattern matched, clean up and return NULL */
    if (pattern_found == 0)
    {
        calceph_lexer_destroy(cplexer);
        return NULL;
    }

    return cplexer;
}

/*--------------------------------------------------------------------------*/
/*! Preprocess tokens for other_calendar: quoted years, special markers
 *  ([e], [w], [N], [Z], [s]), ie, -> Ye
 *
 *  @param lexer    (in/out)    lexer to preprocess
 */
/*--------------------------------------------------------------------------*/
static void other_calendar_preprocess_tokens(struct lexer *lexer)
{
    struct token *tok, *tnext;

    /* i -> Y for 2-digit integers preceded by a quote */
    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        tnext = tok->next;
        if (tok->type == TOKEN_QUOTE && tnext != NULL &&
            tnext->type == TOKEN_INTEGER && (tnext->end - tnext->start) == 1)
        {
            tok->type = TOKEN_YEAR;
            tok->end = tnext->end;
            calceph_lexer_remove_token(lexer, tnext);
        }
    }

    /* Replace special markers */
    calceph_lexer_replace_tokens_from_pattern(lexer, "[e]", "*e*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "[w]", "*w*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "[N]", "*N*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "[Z]", "*Z*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "[s]", "*s*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "ie,", "Ye*", 1);
}

/*--------------------------------------------------------------------------*/
/*! Parse era, weekday, AM/PM, timezone, and timescale from tokens for other_calendar
 *
 *  @return 0 on success, -1 on error
 *
 *  @param lexer     (in/out)  Lexer containing the tokenized input string
 *  @param timestr   (in)      Original input string (used to extract substrings)
 *  @param timespec  (in/out)  Structure receiving parsed specifiers (era, am/pm, timezone)
 *  @param time      (in/out)  Structure receiving the parsed time system
 */
/*--------------------------------------------------------------------------*/
static int other_calendar_parse_time_specifiers(struct lexer *lexer, const char *timestr,
                                                struct time_specifiers *timespec, struct calceph_time *time)
{
    struct token *tok;
    t_calcephcharvalue tvalue;
    int era = 0, weekday = 0, ampm = 0;
    int timezonefound = 0, timescalefound = 0;

    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        switch (tok->type)
        {
            case TOKEN_ERA:
                if (era != 0)
                    return -1;
                calceph_token_read(tok, timestr, tvalue);
                timespec->era = (strcmp(tvalue, "BC") == 0 || strcmp(tvalue, "B.C.") == 0) ? 1 : 2;
                era = 1;
                break;

            case TOKEN_WEEKDAY:
                if (weekday != 0)
                    return -1;
                weekday = get_weekday_number(tok, timestr);
                break;

            case TOKEN_AMPM:
                if (ampm != 0)
                    return -1;
                calceph_token_read(tok, timestr, tvalue);
                timespec->ampm = (strcmp(tvalue, "AM") == 0 || strcmp(tvalue, "A.M.") == 0) ? 1 : 2;
                ampm = 1;
                break;

            case TOKEN_TIMEZONE:
                if (timezonefound)
                    return -1;
                calceph_token_read(tok, timestr, timespec->timezone);
                timezonefound = 1;
                break;

            case TOKEN_TIMESCALE:
                if (timescalefound)
                    return -1;
                calceph_token_read(tok, timestr, tvalue);
                if (strcmp(tvalue, "UTC") == 0)
                    time->timescale = CALCEPH_UTC;
                else if (strcmp(tvalue, "TDT") == 0)
                    time->timescale = CALCEPH_TT;
                else if (strcmp(tvalue, "TCB") == 0)
                    time->timescale = CALCEPH_TCB;
                else if (strcmp(tvalue, "TDB") == 0)
                    time->timescale = CALCEPH_TDB;
                else if (strcmp(tvalue, "TAI") == 0)
                    time->timescale = CALCEPH_TAI;
                else if (strcmp(tvalue, "TT") == 0)
                    time->timescale = CALCEPH_TT;
                else
                    return -1;
                timescalefound = 1;
                break;
            default:
                break;
        }
    }

    /* Remove specifiers */
    calceph_lexer_replace_tokens_from_pattern(lexer, "e", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "w", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "N", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Z", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "s", "*", 1);

    return 0;
}

/*--------------------------------------------------------------------------*/
/*! Check for redundant punctuation tokens for other_calendar
 *
 *  @return 1 if no redundancy, 0 if error
 *
 *  @param lexer    (in/out)  Lexer containing the tokenized input string
 */
/*--------------------------------------------------------------------------*/
static int other_calendar_check_token_redundance(struct lexer *lexer)
{
    struct token *tok;

    enum tokentype toktype[7] = { TOKEN_QUOTE, TOKEN_COMMA, TOKEN_DASH, TOKEN_DOT,
        TOKEN_SLASH, TOKEN_COLON, TOKEN_DOY_MARKER
    };
    int j, k;

    for (tok = lexer->head; tok != NULL; tok = tok->next)
        for (j = 0; j < 7; j++)
            for (k = 0; k < 7; k++)
                if (tok->type == toktype[j] && tok->next != NULL && tok->next->type == toktype[k])
                    return 0;

    return 1;
}

/*--------------------------------------------------------------------------*/
/*! Parse calendar date/time formats that are not ISO
 *
 * @return 1 if parsing succeeds, 0 otherwise
 *
 * @param lexer     (in/out)    The lexer containing tokenized input
 * @param timestr   (in)        Original input string
 * @param time      (out)       Parsed time structure
 */
/*--------------------------------------------------------------------------*/
static int parse_other_calendar_formats(struct lexer *lexer, const char *timestr, struct calceph_time *time,
                                        struct time_specifiers *timespec)
{
    struct token *tok;

    t_calcephcharvalue tvalue;

    t_calcephcharvalue transformation_pattern;

    double dayofyear = 0., day = 0., hour = 0., minute = 0., second = 0.;

    other_calendar_preprocess_tokens(lexer);

    if (other_calendar_parse_time_specifiers(lexer, timestr, timespec, time) != 0)
        return 0;

    if (other_calendar_check_token_redundance(lexer) == 0)
        return -1;

    struct lexer *newlexer = other_calendar_find_transformation(lexer, transformation_pattern);

    int iscorrect = 0;

    if (newlexer != NULL)
    {
        iscorrect = 1;
        calceph_lexer_replace(lexer, newlexer);
    }

    if (iscorrect == 1)
    {
        time->datatype = CALCEPH_CALENDAR;
        for (tok = lexer->head; tok != NULL; tok = tok->next)
        {
            calceph_token_read(tok, timestr, tvalue);
            switch (tok->type)
            {
                case TOKEN_YEAR:
                    if (tvalue[0] == '\'')
                    {
                        time->datetime.calendar.year = atoi(tvalue + 1);
                    }
                    else
                    {
                        time->datetime.calendar.year = atoi(tvalue);
                    }
                    if (tvalue[0] == '\'' || time->datetime.calendar.year < 100)
                    {
                        if (time->datetime.calendar.year > 68 && timespec->era == 0)
                            time->datetime.calendar.year += 1900;
                        else if (timespec->era == 0)
                            time->datetime.calendar.year += 2000;
                    }
                    break;
                case TOKEN_MONTH:
                    time->datetime.calendar.month = get_month_number(tok, timestr);
                    break;
                case TOKEN_DAY_OF_MONTH:
                    day = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                    break;
                case TOKEN_DAY_OF_YEAR:
                    dayofyear = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                    break;
                case TOKEN_HOUR:
                    hour = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                    break;
                case TOKEN_MINUTE:
                    minute = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                    break;
                case TOKEN_SECOND:
                    second = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                    break;
                default:
                    return 0;
            }
        }

        compute_decimal_time(time, dayofyear, day, hour, minute, second);
    }

    return iscorrect;
}

/*--------------------------------------------------------------------------*/
/*! Parse date/time strings using fallback rules (last chance parser).
 *
 *  This function is invoked after the ISO format (`parse_iso`) and other
 *  calendar formats (`parse_other_calendar_formats`) have failed. It tries
 *  to normalize and parse ambiguous or less common formats.
 *
 *  @return  1  if parsing succeeds
 *  @return  0  if tokens remain unrecognized
 *  @return -1  if an error occurs
 *
 *  @param (in/out) lexer   the lexer with tokenized input
 *  @param (in)     timestr original input string
 *  @param (out)    time    parsed time structure
 *
 */
/*--------------------------------------------------------------------------*/
static int parse_last_rules(struct lexer *lexer, const char *timestr, struct calceph_time *time,
                            struct time_specifiers *timespec)
{
    int ret = 0;

    struct token *tok;

    int hastokyear = 0;

    t_calcephcharvalue tvalue;

    /* remove all ',', '-' and '/' */
    calceph_lexer_replace_tokens_from_pattern(lexer, ",", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "-", "*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "/", "*", 1);

    ret = parse_other_calendar_formats(lexer, timestr, time, timespec);

    if (ret == 1)
    {
        return 1;
    }

    /* transform integer and float tokens in time */

    calceph_lexer_replace_tokens_from_pattern(lexer, "i:i:i:n", "D*H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i:i:i:i", "D*H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i:i:n", "H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i:i:i", "H*M*S", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i:n", "H*M", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "i:i", "H*M", 1);

    /* remove all ':' */
    calceph_lexer_replace_tokens_from_pattern(lexer, ":", "*", 1);

    /* last transformations */
    calceph_lexer_replace_tokens_from_pattern(lexer, "miiH", "mDY<", 2);
    calceph_lexer_replace_tokens_from_pattern(lexer, "mi", "mD", 2);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Siim", "SYDm", 3);
    calceph_lexer_replace_tokens_from_pattern(lexer, "im", "Dm", 3);
    calceph_lexer_replace_tokens_from_pattern(lexer, "miY", "mDY", 3);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Ymi", "YmD", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Smi", "SmD", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Mmi", "MmD", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "imY", "DmY", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "imH", "DmH", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Yid", "Yy*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "iYd", "yY*", 1);
    calceph_lexer_replace_tokens_from_pattern(lexer, "Ydi", "Y*y", 1);

    /* if 'integer' or 'number' tokens are present at this state : error */
    if (calceph_lexer_replace_tokens_from_pattern(lexer, "i", "i", 1) > 0 ||
        calceph_lexer_replace_tokens_from_pattern(lexer, "n", "n", 1) > 0)
    {
        return -1;
    }

    double day = -1., dayofyear = -1., hour = -1., minute = -1., second = -1.;

    time->datatype = CALCEPH_CALENDAR;
    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        calceph_token_read(tok, timestr, tvalue);
        switch (tok->type)
        {
            case TOKEN_YEAR:
                if (hastokyear != 0)
                    return -1;
                hastokyear = 1;
                time->datetime.calendar.year = atoi(tvalue);
                break;
            case TOKEN_MONTH:
                if (time->datetime.calendar.month != 0)
                    return -1;
                time->datetime.calendar.month = get_month_number(tok, timestr);
                break;
            case TOKEN_DAY_OF_MONTH:
                if (day != -1.)
                    return -1;
                day = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_DAY_OF_YEAR:
                if (dayofyear != -1.)
                    return -1;
                dayofyear = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_HOUR:
                if (hour != -1.)
                    return -1;
                hour = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_MINUTE:
                if (minute != -1.)
                    return -1;
                minute = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                break;
            case TOKEN_SECOND:
                if (second != -1.)
                    return -1;
                second = calceph_strtod(tvalue, NULL, *lexer->ephlocale);
                break;
            default:
                return 0;
        }
    }

    if (hastokyear == 0 || ((time->datetime.calendar.month == 0 || day == -1.) && dayofyear == -1.))
        return 0;

    if (dayofyear == -1.)
        dayofyear = 0.;
    if (day == -1.)
        day = 0.;
    if (hour == -1.)
        hour = 0.;
    if (minute == -1.)
        minute = 0.;
    if (second == -1.)
        second = 0.;

    compute_decimal_time(time, dayofyear, day, hour, minute, second);

    return 1;
}

static int is_leap_year(int year)
{
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

static int is_valid_day(int year, int month, int day)
{
    if (month < 1 || month > 12)
        return 0;

    int days_in_month[] = {
        31,
        28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31
    };

    if (month == 2 && is_leap_year(year))
        days_in_month[1] = 29;

    return day >= 1 && day <= days_in_month[month - 1] ? 1 : 0;
}

/*-----------------------------------------------------------------*/
/* principal parsing function */
/*-----------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*! Parse a time string into a calceph_time structure.
 *
 *  This function tokenizes and interprets a given time string
 *  according to supported calendar and time formats:
 *   1. Julian Date (JD)
 *   2. ISO formats
 *   3. Other calendar-based formats
 *   4. Fallback heuristic transformations
 *
 *  It allocates and fills a new calceph_time structure upon success.
 *
 *  @return 0 on success, 1 on error 
 *
 *  @param (in)  timestr    String representation of the time to parse.
 *  @param (out) time       Pointer to the resulting calceph_time structure.
 *
 *  @note The default timescale is initialized to CALCEPH_UTC.
 */
/*--------------------------------------------------------------------------*/
int calceph_parse_time(struct calceph_locale *pclocale, const char *timestr, struct calceph_time *time)
{

    /* Common timezone abbreviations -> UTC offsets */
    const char *timezones[8][2] = {
        {"EST", "UTC-5:00"}, {"CST", "UTC-6:00"},
        {"MST", "UTC-7:00"}, {"PST", "UTC-8:00"},
        {"EDT", "UTC-4:00"}, {"CDT", "UTC-5:00"},
        {"MDT", "UTC-6:00"}, {"PDT", "UTC-7:00"}
    };

    if (timestr == NULL || *timestr == '\0')
    {
/* GCOVR_EXCL_START */
        fatalerror("calceph_parse_time: empty or NULL time string.\n");
/* GCOVR_EXCL_STOP */
        return 1;
    }

    /* Allow optional '@' prefix (used in some date notations) */
    if (*timestr == '@')
        timestr++;

    /* Tokenize input string */
    struct lexer *lexer = calceph_time_tokenize(pclocale, timestr);

    if (lexer == NULL)
    {
        fatalerror("The format time string is not supported. The parsing processing fails\n");
        return 1;
    }

    /* Allocate parsing result structure */
    struct time_specifiers *timespec = calloc(1, sizeof(struct time_specifiers));

    if (timespec == NULL)
    {
        calceph_lexer_destroy(lexer);
        free(timespec);
        fatalerror("Memory allocation failed in calceph_parse_time.\n");
        return 1;
    }

    time->timescale = CALCEPH_UTC;

    initial_token_processing(lexer);

    /*------------------------------------------------------------------*/
    /* Step 1: Try Julian Date (JD) format                              */
    /*------------------------------------------------------------------*/
    time->datetime.juliandate.integerpart = 0.;
    time->datetime.juliandate.decimalpart = 0.;

    int isjd = parse_jd(lexer, timestr, time);

    if (isjd != 0)
    {
        calceph_lexer_destroy(lexer);
        free(timespec);
        if (isjd == -1)
        {
            fatalerror("Julian Date parsing failed.\n");
            return 1;
        }

        return 0;
    }

    /*------------------------------------------------------------------*/
    /* Step 2: Try ISO format                                           */
    /*------------------------------------------------------------------*/
    time->datetime.calendar.year = 0;
    time->datetime.calendar.month = 0;
    time->datetime.calendar.day = 0;
    time->datetime.calendar.hour = 0;
    time->datetime.calendar.minute = 0;
    time->datetime.calendar.second = 0.;

    int isiso = parse_iso(lexer, timestr, time);

    if (isiso == -1)
    {
        calceph_lexer_destroy(lexer);
        free(timespec);
        fatalerror("ISO date parsing failed.\n");
        return 1;
    }

    /*------------------------------------------------------------------*/
    /* Step 3: Try other calendar-based formats                         */
    /*------------------------------------------------------------------*/
    int isothercalendar = 0;

    if (isiso == 0)
    {
        isothercalendar = parse_other_calendar_formats(lexer, timestr, time, timespec);
        if (isothercalendar == -1)
        {
            calceph_lexer_destroy(lexer);
            free(timespec);
            fatalerror("Other calendar format parsing failed.\n");
            return 1;
        }
    }

    /*------------------------------------------------------------------*/
    /* Step 4: Fallback parsing rules (heuristics)                      */
    /*------------------------------------------------------------------*/
    int islastrules = 0;

    if (isiso == 0 && isothercalendar == 0)
    {
        islastrules = parse_last_rules(lexer, timestr, time, timespec);
        if (islastrules != 1)
        {
            calceph_lexer_destroy(lexer);
            free(timespec);
            fatalerror("Fallback rule parsing failed.\n");
            return 1;
        }
    }

    calceph_lexer_destroy(lexer);

    /*------------------------------------------------------------------*/
    /* Step 5: Post-processing (era, AM/PM, timezone adjustments)       */
    /*------------------------------------------------------------------*/

    /* Handle BC era */
    if (timespec->era == 1)
        time->datetime.calendar.year = -time->datetime.calendar.year + 1;

    /* Handle AM/PM corrections */
    if (timespec->ampm == 1 && time->datetime.calendar.hour >= 12)
        time->datetime.calendar.hour -= 12;
    else if (timespec->ampm == 2 && time->datetime.calendar.hour < 12)
        time->datetime.calendar.hour += 12;

    /* Normalize timezone */
    calceph_strtoupper(timespec->timezone);

    int j;

    for (j = 0; j < 8; j++)
    {
        if (strncmp(timezones[j][0], timespec->timezone, 3) == 0)
        {
            strncpy(timespec->timezone, timezones[j][1], 8);
            break;
        }
    }

    /*------------------------------------------------------------------*/
    /* Step 6: Apply UTC offset from timezone                           */
    /*------------------------------------------------------------------*/
    char *p = timespec->timezone;
    int utchour = 0, utcminute = 0;

    if (strncmp(p, "UTC-", 4) == 0 || strncmp(p, "UTC+", 4) == 0)
    {
        char sign = p[3];

        p += 4;

        /* Parse hours */
        while (*p != '\0' && *p != ':')
        {
            utchour = utchour * 10 + (*p - '0');
            p++;
        }

        /* Parse minutes if present */
        if (*p == ':')
        {
            p++;
            while (*p != '\0')
            {
                utcminute = utcminute * 10 + (*p - '0');
                p++;
            }
        }

        if (utcminute >= 60 || utchour >= 13)
        {
            free(timespec);
            fatalerror("Invalid UTC offset in timezone.\n");
            return 1;
        }

        /* Apply offset */
        if (sign == '-')
        {
            time->datetime.calendar.hour += utchour;
            time->datetime.calendar.minute += utcminute;
            if (time->datetime.calendar.minute >= 60)
            {
                time->datetime.calendar.hour++;
                time->datetime.calendar.minute -= 60;
            }
            if (time->datetime.calendar.hour >= 24)
            {
                time->datetime.calendar.day++;
                time->datetime.calendar.hour -= 24;

                int days_in_month[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

                if (is_leap_year(time->datetime.calendar.year))
                {
                    days_in_month[1] = 29;
                }

                if (time->datetime.calendar.day > days_in_month[time->datetime.calendar.month - 1])
                {
                    time->datetime.calendar.day = 1;
                    time->datetime.calendar.month++;

                    if (time->datetime.calendar.month > 12)
                    {
                        time->datetime.calendar.month = 1;
                        time->datetime.calendar.year++;
                    }
                }
            }
        }
        else
        {
            /* sign == '+' */
            time->datetime.calendar.hour -= utchour;
            time->datetime.calendar.minute -= utcminute;
            if (time->datetime.calendar.minute < 0)
            {
                time->datetime.calendar.hour--;
                time->datetime.calendar.minute += 60;
            }
            if (time->datetime.calendar.hour < 0)
            {
                time->datetime.calendar.day--;
                time->datetime.calendar.hour += 24;

                if (time->datetime.calendar.day < 1)
                {
                    time->datetime.calendar.month--;
                    if (time->datetime.calendar.month < 1)
                    {
                        time->datetime.calendar.month = 12;
                        time->datetime.calendar.year--;
                    }

                    int days_in_prev_month[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

                    if (is_leap_year(time->datetime.calendar.year) && time->datetime.calendar.month == 2)
                    {
                        days_in_prev_month[1] = 29;
                    }

                    time->datetime.calendar.day = days_in_prev_month[time->datetime.calendar.month - 1];
                }
            }
        }
    }

    free(timespec);

    /*------------------------------------------------------------------*/
    /* Verify bounds of values in the result */
    /*------------------------------------------------------------------*/
    if (time->datetime.calendar.month < 1 || time->datetime.calendar.month > 12 ||
        time->datetime.calendar.hour < 0 || time->datetime.calendar.hour >= 24 ||
        time->datetime.calendar.minute < 0 || time->datetime.calendar.minute >= 60 ||
        time->datetime.calendar.second < 0. || time->datetime.calendar.second >= 60. ||
        is_valid_day(time->datetime.calendar.year, time->datetime.calendar.month, time->datetime.calendar.day) == 0)
    {
        fatalerror("Values in the result out of bounds.\n");
        return 1;
    }

    return 0;
}
