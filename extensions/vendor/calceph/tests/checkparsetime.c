/*-----------------------------------------------------------------*/
/*!
  \file checkparsetime.c
  \brief Check that the results of calceph_parse_time are correct.

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
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#include "calceph.h"
#include "openfiles.h"

#if HAVE_MATH_H
/* enable M_PI with windows sdk */
#define _USE_MATH_DEFINES
#include <math.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif

#include "real.h"
#include "calcephinternal.h"
#include "util.h"

#define MAXLINE 128

static void hidemsg(const char *msg);
int main(void);

static void hidemsg(const char *PARAMETER_UNUSED(msg))
{
#if HAVE_PRAGMA_UNUSED
#pragma unused(msg)
#endif
    /*printf("msg='%s'\n", msg) */
}

struct test_calendar
{
    int test_number;
    t_calcephcharvalue timestr;
    int year;
    int month;
    int day;
    int hour;
    int minute;
    double second;
    int ts;
};

static void print_jd_result(char *timestr, struct calceph_time *time, double jdint, double jddecimal, int ts)
{
    printf("+------------------------------------------------------+\n");
    printf("| TEST (JD): %-42s|\n", timestr);
    printf("+------------------+-----------------+-----------------+\n");
    if (time == NULL)
    {
        printf("|                    RESULT is NULL                    |\n");
        printf("+------------------+-----------------+-----------------+\n");
        return;
    }
    printf("|                  |    EXPECTED     |     CURRENT     |\n");
    printf("+------------------+-----------------+-----------------+\n");
    printf("|    DATATYPE      |        %i        |        %i        |\n", 1, time->datatype);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|   JULIAN DATE    | %-15.6f | %-15.6f |\n", jdint, time->datetime.juliandate.integerpart);
    printf("|                  | %-15.6f | %-15.6f |\n", jddecimal, time->datetime.juliandate.decimalpart);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|    TIMESYSTEM    |        %i        |        %i        |\n", ts, time->timescale);
    printf("+------------------+-----------------+-----------------+\n");
}

static void print_calendar_result(int test_number, char *timestr, struct calceph_time *time, int year, int month, int day,
                           int hour, int minute, double second, int ts)
{
    printf("+------------------------------------------------------+\n");
    printf("| TEST (CALENDAR-%-3i) : %-31s|\n", test_number, timestr);
    printf("+------------------+-----------------+-----------------+\n");
    if (time == NULL)
    {
        printf("|                    RESULT is NULL                    |\n");
        printf("+------------------+-----------------+-----------------+\n");
        return;
    }
    printf("|                  |    EXPECTED     |     CURRENT     |\n");
    printf("+------------------+-----------------+-----------------+\n");
    printf("|     DATATYPE     |       %-5i     |       %-5i     |\n", 2, time->datatype);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|       YEAR       |      %-5i      |      %-5i      |\n", year, time->datetime.calendar.year);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|      MONTH       |       %-5i     |       %-5i     |\n", month, time->datetime.calendar.month);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|       DAY        |       %-5i     |       %-5i     |\n", day, time->datetime.calendar.day);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|       HOUR       |       %-5i     |       %-5i     |\n", hour, time->datetime.calendar.hour);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|      MINUTE      |       %-5i     |       %-5i     |\n", minute, time->datetime.calendar.minute);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|      SECOND      |    %-12.6f |    %-12.6f |\n", second, time->datetime.calendar.second);
    printf("+------------------+-----------------+-----------------+\n");
    printf("|    TIMESYSTEM    |        %i        |        %i        |\n", ts, time->timescale);
    printf("+------------------+-----------------+-----------------+\n");
}

static int test_parse_jd(struct calceph_locale *clocale, char *timestr, double jdint, double jddecimal, int ts)
{
    int ret = 0;

    struct calceph_time time;

    if (calceph_parse_time(clocale, timestr, &time) == 1 || time.datatype != CALCEPH_JULIANDATE)
    {
        ret = 1;
    }
    else if (fabs(time.datetime.juliandate.integerpart - jdint) > 1e-6)
    {
        ret = 1;
    }
    else if (fabs(time.datetime.juliandate.decimalpart - jddecimal) > 1e-6)
    {
        ret = 1;
    }
    else if (ts != time.timescale)
    {
        ret = 1;
    }

    if (ret == 1)
    {
        print_jd_result(timestr, &time, jdint, jddecimal, ts);
    }

    return ret;
}

static int test_parse_calendar(struct calceph_locale *clocale, struct test_calendar test)
{
    int ret = 0;

    struct calceph_time time;

    if (calceph_parse_time(clocale, test.timestr, &time) == 1 || time.datatype != CALCEPH_CALENDAR)
    {
        ret = 1;
    }
    else if (time.datetime.calendar.year != test.year ||
             time.datetime.calendar.month != test.month ||
             time.datetime.calendar.day != test.day ||
             time.datetime.calendar.hour != test.hour || time.datetime.calendar.minute != test.minute)
    {
        ret = 1;
    }
    else if (fabs(time.datetime.calendar.second - test.second) > 1e-6)
    {
        ret = 1;
    }
    else if (test.ts != time.timescale)
    {
        ret = 1;
    }

    if (ret == 1)
    {
        print_calendar_result(test.test_number, test.timestr, &time, test.year, test.month, test.day, test.hour,
                              test.minute, test.second, test.ts);
    }

    return ret;
}

static int test_on_error(struct calceph_locale *clocale, char *timestr)
{
    int ret = 0;

    struct calceph_time time;

    if (calceph_parse_time(clocale, timestr, &time) == 0)
    {
        printf("Test '%s' should return 1, but 0 was returned\n", timestr);
        print_calendar_result(1111, timestr, &time, 0, 0, 0, 0, 0, 0, 0);
        ret = 1;
    }

    return ret;
}

/*-----------------------------------------------------------------*/
/* main program */
/*-----------------------------------------------------------------*/
int main(void)
{
    calceph_seterrorhandler(3, hidemsg);

    int res = 0;

    int test_number = 1;

    t_calcephcharvalue line, timestr, timesysstr, transformation;

    int Y, m, D, M, H;

    double S;

    char *token;

    int ts;

    struct calceph_locale clocale;

    FILE *file = tests_calceph_open_r("datetests.csv");

    if (file == NULL)
    {
        puts("the file tests_date.csv can not be opened");
        return 1;
    }

    calceph_locale_init(&clocale);   

    /****************************************/
    /* TEST JD                              */
    /****************************************/ 
    if (test_parse_jd(&clocale, "JD 2433282.529", 2433282., 0.529, CALCEPH_UTC) == 1)
        res = 1;

    if (test_parse_jd(&clocale, "JD 2433282.529 (TDB)", 2433282., 0.529, CALCEPH_TDB) == 1)
        res = 1;

    if (test_parse_jd(&clocale, "jd 2433282.529 (UTC)", 2433282., 0.529, CALCEPH_UTC) == 1)
        res = 1;

    if (test_parse_jd(&clocale, "2433282.529 (JD)", 2433282., 0.529, CALCEPH_UTC) == 1)
        res = 1;

    if (test_parse_jd(&clocale, "2433282.529 JD", 2433282., 0.529, CALCEPH_UTC) == 1)
        res = 1;

    /****************************************/
    /* TEST ERROR                           */
    /****************************************/
    if (test_on_error(&clocale, "1987-102T16:31:12.814 azerty") == 1)
        res = 1;

    if (test_on_error(&clocale, "@123") == 1)
        res = 1;

    if (test_on_error(&clocale, "Mar 29,") == 1)
        res = 1;

    if (test_on_error(&clocale, "11 May JUN 1990 9:00:00") == 1)
        res = 1;

    if (test_on_error(&clocale, "2025 Oct 9 25:00:02") == 1)
        res = 1;

    if (test_on_error(&clocale, "2025 Oct 9 11:70:01") == 1)
        res = 1;

    if (test_on_error(&clocale, "2020 MAR 32") == 1)
        res = 1;

    if (test_on_error(&clocale, "  ") == 1)
        res = 1;

    /****************************************/
    /* TEST CALENDAR                        */
    /****************************************/
    if (fgets(line, MAXLINE, file) == NULL)
    {
        fclose(file);
        return 0;
    }

    while (fgets(line, MAXLINE, file) != NULL)
    {
        line[strcspn(line, "\r\n")] = 0;

        token = strtok(line, ";");
        if (!token)
            continue;
        calceph_snprintf(timestr, sizeof(timestr), "%s", token);

        token = strtok(NULL, ";");
        if (!token)
            continue;
        calceph_snprintf(transformation, sizeof(transformation), "%s", token);

        token = strtok(NULL, ";");
        Y = token ? atoi(token) : 0;

        token = strtok(NULL, ";");
        m = token ? atoi(token) : 0;

        token = strtok(NULL, ";");
        D = token ? atoi(token) : 0;

        token = strtok(NULL, ";");
        H = token ? atoi(token) : 0;

        token = strtok(NULL, ";");
        M = token ? atoi(token) : 0;

        token = strtok(NULL, ";");
        S = token ? atof(token) : 0.0;

        token = strtok(NULL, ";");
        strcpy(timesysstr, token ? token : "");
        if (strcmp(timesysstr, "TDB") == 0)
        {
            ts = CALCEPH_TDB;
        }
        else if (strcmp(timesysstr, "TT") == 0)
        {
            ts = CALCEPH_TT;
        }
        else if (strcmp(timesysstr, "TCB") == 0)
        {
            ts = CALCEPH_TCB;
        }
        else if (strcmp(timesysstr, "TAI") == 0)
        {
            ts = CALCEPH_TAI;
        }
        else
        {
            ts = CALCEPH_UTC;
        }

        struct test_calendar test_cal = {
            test_number,
            "",
            Y,
            m,
            D,
            H,
            M,
            S,
            ts
        };

        calceph_snprintf(test_cal.timestr, sizeof(test_cal.timestr), "%s", timestr);

        if (test_parse_calendar(&clocale, test_cal) == 1)
            res = 1;

        test_number++;
    }

    calceph_locale_clear(&clocale);   

    fclose(file);

    return res;
}
