/*-----------------------------------------------------------------*/
/*!
  \file calcephtimelexer.c
  \brief internal functions for lexing datetime strings

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
#include <ctype.h>
#define __CALCEPH_WITHIN_CALCEPH 1
#include "calceph.h"
#include "real.h"
#include "util.h"
#include "calcephinternal.h"
#include "calcephtimelexer.h"

/*-----------------------------------------------------------------*/
/* static functions */
/*-----------------------------------------------------------------*/

static char calceph_time_map_token_to_char(enum tokentype t);
static enum tokentype calceph_time_map_char_to_token(char c);
static int match_keywords(const char *timestr, int curpos, const char *const *keywords, int n, int mode);
static int isweekday(const char *timestr, int curpos);
static int ismonth(const char *timestr, int curpos);
static int istimezone(const char *timestr, int curpos);
static int istimesystem(const char *timestr, int curpos);
static int iseraspecifier(const char *timestr, int curpos);
static int isampmspecifier(const char *timestr, int curpos);

/*-----------------------------------------------------------------*/
/* struct */
/*-----------------------------------------------------------------*/
struct simple_token_map
{
    char c;
    int type;
};

/*--------------------------------------------------------------------------*/
/*! Initialize a new allocated token structure
 *
 *  @return pointer to the newly created token (NULL if allocation fails)
 *
 *  @param type   (in) token type
 *  @param start  (in) start index in the source string
 *  @param end    (in) end index in the source string
 */
/*--------------------------------------------------------------------------*/
struct token *calceph_token_init(enum tokentype type, int start, int end)
{
    struct token *token = calloc(1, sizeof(struct token));

    if (!token)
        return NULL;

    token->type = type;
    token->start = start;
    token->end = end;
    return token;
}

/*--------------------------------------------------------------------------*/
/*! Create a deep copy of a token
 *
 *  @return pointer to the duplicated token (NULL on failure)
 *
 *  @param token (in) token to copy
 */
/*--------------------------------------------------------------------------*/
struct token *calceph_token_copy(struct token *token)
{
    struct token *cp_token = calloc(1, sizeof(struct token));

    if (!cp_token)
        return NULL;

    cp_token->type = token->type;
    cp_token->start = token->start;
    cp_token->end = token->end;

    return cp_token;
}

/*--------------------------------------------------------------------------*/
/*! Extract substring corresponding to a token
 *
 *  @param token   (in)  token descriptor
 *  @param timestr (in)  source string
 *  @param res     (out) buffer to store the extracted substring
 */
/*--------------------------------------------------------------------------*/
void calceph_token_read(struct token *token, const char *timestr, t_calcephcharvalue res)
{
    int len;

    if (token != NULL && timestr != NULL)
    {
        len = token->end - token->start + 1;
        memcpy(res, timestr + token->start, len);
        res[len] = '\0';
    }
}

/*--------------------------------------------------------------------------*/
/*! Initialize a lexer structure
 *
 *  @return pointer to the newly created lexer (NULL if allocation fails)
 */
/*--------------------------------------------------------------------------*/
struct lexer *calceph_lexer_init(struct calceph_locale *pclocale)
{
    struct lexer *lexer = calloc(1, sizeof(struct lexer));

    if (lexer != NULL)
    {
        lexer->head = NULL;
        lexer->tail = NULL;
        lexer->ephlocale = pclocale;
        lexer->ntokens = 0;
    }

    return lexer;
}

/*--------------------------------------------------------------------------*/
/*! Create a deep copy of a lexer and all its tokens.
 *
 *  @return pointer to the duplicated lexer (NULL on failure)
 *
 *  @param  lexer (in)  lexer to copy
 */
/*--------------------------------------------------------------------------*/
struct lexer *calceph_lexer_copy(struct lexer *lexer)
{
    struct lexer *cplexer = calloc(1, sizeof(struct lexer));

    if (cplexer == NULL)
        return NULL;

    cplexer->ntokens = lexer->ntokens;

    struct token *cptok, *tok;

    for (tok = lexer->head; tok != NULL; tok = tok->next)
    {
        cptok = calceph_token_copy(tok);

        if (cptok == NULL)
        {
            free(cplexer);
            return NULL;
        }

        calceph_lexer_add_token(cplexer, cptok);
    }

    return cplexer;
}

/*--------------------------------------------------------------------------*/
/*! Remove and free all tokens from a lexer (but not the lexer itself).
 *
 *  @param lexer (in/out) lexer whose tokens will be freed
 */
/*--------------------------------------------------------------------------*/
void calceph_lexer_clear(struct lexer *lexer)
{
    struct token *t, *next;

    if (lexer != NULL)
    {
        t = lexer->head;
        while (t != NULL)
        {
            next = t->next;
            free(t);
            t = next;
        }
        lexer->head = NULL;
        lexer->tail = NULL;
        lexer->ntokens = 0;
    }
}

/*--------------------------------------------------------------------------*/
/*! Destroy a lexer and its tokens
 *
 *  Frees all tokens contained in the lexer, then the lexer itself
 *
 *  @param lexer (in) lexer object to free
 */
/*--------------------------------------------------------------------------*/
void calceph_lexer_destroy(struct lexer *lexer)
{
    calceph_lexer_clear(lexer);

    if (lexer != NULL)
    {
        free(lexer);
    }
}

/*--------------------------------------------------------------------------*/
/*! Replace the contents of one lexer with another and free the source lexer.
 *
 *  @param lexer  (out) destination lexer to fill
 *  @param rlexer (in)  source lexer to copy and destroy
 */
/*--------------------------------------------------------------------------*/
void calceph_lexer_replace(struct lexer *lexer, struct lexer *rlexer)
{
/* GCOVR_EXCL_START */
    if (lexer == NULL || rlexer == NULL)
        return;
/* GCOVR_EXCL_STOP */

    calceph_lexer_clear(lexer);

    lexer->ntokens = rlexer->ntokens;
    lexer->head = rlexer->head;
    lexer->tail = rlexer->tail;

    free(rlexer);
}

/*--------------------------------------------------------------------------*/
/*! Add a token to the end of lexer 
 *
 *  @return total number of tokens in the lexer after insertion
 *  @return 0 if lexer or token is NULL
 *
 *  @param lexer (in/out) lexer object
 *  @param token (in)     token to add
 */
/*--------------------------------------------------------------------------*/
int calceph_lexer_add_token(struct lexer *lexer, struct token *token)
{
    if (lexer == NULL || token == NULL)
        return 0;

    if (lexer->head == NULL)
        lexer->head = token;
    else
    {
        token->prec = lexer->tail;
        lexer->tail->next = token;
    }
    lexer->tail = token;

    return ++lexer->ntokens;
}

/*--------------------------------------------------------------------------*/
/*! Remove a token from the lexer
 *
 *  @return updated number of tokens in the lexer
 *  @return -1 if lexer is NULL, token is NULL, or token not found
 *
 *  @param lexer (in/out) lexer object
 *  @param token (in)     token to remove
 */
/*--------------------------------------------------------------------------*/
int calceph_lexer_remove_token(struct lexer *lexer, struct token *token)
{
    if (lexer == NULL || token == NULL)
        return -1;

    struct token *t;

    for (t = lexer->head; t; t = t->next)
    {
        if (t == token)
        {
            if (t == lexer->head)
                lexer->head = t->next;
            if (t == lexer->tail)
                lexer->tail = t->prec;
            if (t->prec)
                t->prec->next = t->next;
            if (t->next)
                t->next->prec = t->prec;

            free(t);
            return --lexer->ntokens;
        }
    }

    return -1;                  /* token not found */
}

/*--------------------------------------------------------------------------*/
/*! Retrieve a token by index
 *
 *  @return pointer to the token at the requested index
 *  @return NULL if lexer is NULL or index is invalid
 *
 *  @param lexer (in) lexer object
 *  @param index (in) zero-based position of the token
 */
/*--------------------------------------------------------------------------*/
struct token *calceph_lexer_get_token_at_index(struct lexer *lexer, int index)
{
    if (lexer == NULL || index < 0)
        return NULL;

    struct token *tok = lexer->head;
    int j;

    for (j = 0; j < index && tok != NULL; j++)
        tok = tok->next;

    return tok;
}

/*--------------------------------------------------------------------------*/
/*! Replace tokens in the lexer based on a pattern
 *
 *  The behavior depends on the mode:
 *    - mode = 1: search and replace all occurrences
 *    - mode = 2: replace only at the start of the lexer
 *    - mode = 3: replace only at the end of the lexer
 *
 *  Replacement rules (from rpattern):
 *    - '*' : remove the corresponding token
 *    - '<' : merge token with its predecessor (extend predecessor end index)
 *    - other char : replace token type with the mapped type of this char
 *
 *  @return number of occurrences replaced
 *
 *  @param lexer    (in/out) lexer object
 *  @param pattern  (in)     character sequence to search for
 *  @param rpattern (in)     replacement pattern
 *  @param mode     (in)     replacement mode (1=all, 2=start, 3=end)
 */
/*--------------------------------------------------------------------------*/
int calceph_lexer_replace_tokens_from_pattern(struct lexer *lexer, const char *pattern, const char *rpattern, int mode)
{
    t_calcephcharvalue lexerbuffer;

    calceph_lexer_to_string(lexer, lexerbuffer);

    int lenpattern = strlen(pattern);
    int lenlexer = strlen(lexerbuffer);

    int found = 0;

    int j, k;

    int start_index = (mode == 3) ? lenlexer - lenpattern : 0;

    struct token *tok = calceph_lexer_get_token_at_index(lexer, start_index);
    struct token *tnext;

    int flag = 1;

    for (j = start_index; j <= lenlexer - lenpattern && tok != NULL && flag == 1; j++)
    {
        if (mode == 2)
            flag = 0;
        if (strncmp(lexerbuffer + j, pattern, lenpattern) == 0)
        {
            found++;
            for (k = 0; k < lenpattern && tok != NULL; k++)
            {
                tnext = tok->next;
                if (rpattern[k] == '*')
                {
                    calceph_lexer_remove_token(lexer, tok);
                }
                else if (rpattern[k] == '<' && tok->prec != NULL)
                {
                    tok->prec->end = tok->end;
                    calceph_lexer_remove_token(lexer, tok);
                }
                else if (rpattern[k] != pattern[k])
                {
                    tok->type = calceph_time_map_char_to_token(rpattern[k]);
                }
                tok = tnext;
            }
        }
        else
        {
            tok = tok->next;
        }
    }

    return found;
}

/*--------------------------------------------------------------------------*/
/*! Map token type to its corresponding character
 *
 *  @return mapped character for the token type
 *  @return 'x' if the token type is unknown
 *
 *  @param t (in) token type to convert
 */
/*--------------------------------------------------------------------------*/
static char calceph_time_map_token_to_char(enum tokentype t)
{
    switch (t)
    {
        case TOKEN_QUOTE:
            return 'Q';
        case TOKEN_LPAREN:
            return '[';
        case TOKEN_RPAREN:
            return ']';
        case TOKEN_COMMA:
            return ',';
        case TOKEN_DASH:
            return '-';
        case TOKEN_DOT:
            return '.';
        case TOKEN_SLASH:
            return '/';
        case TOKEN_COLON:
            return ':';
        case TOKEN_AMPM:
            return 'N';
        case TOKEN_UTC_PLUS:
            return 'O';
        case TOKEN_TIMEZONE:
            return 'Z';
        case TOKEN_WHITESPACE:
            return 'b';
        case TOKEN_DOY_MARKER:
            return 'd';
        case TOKEN_ERA:
            return 'e';
        case TOKEN_JULIAN:
            return 'j';
        case TOKEN_MONTH:
            return 'm';
        case TOKEN_UTC_MINUS:
            return 'o';
        case TOKEN_TIMESCALE:
            return 's';
        case TOKEN_T_SEPARATOR:
            return 't';
        case TOKEN_WEEKDAY:
            return 'w';
        case TOKEN_INTEGER:
            return 'i';
        case TOKEN_Z:
            return 'z';
        case TOKEN_PLUS:
            return '+';
        case TOKEN_NUMBER:
            return 'n';
        case TOKEN_YEAR:
            return 'Y';
        case TOKEN_DAY_OF_MONTH:
            return 'D';
        case TOKEN_DAY_OF_YEAR:
            return 'y';
        case TOKEN_HOUR:
            return 'H';
        case TOKEN_MINUTE:
            return 'M';
        case TOKEN_SECOND:
            return 'S';
        default:
            return 'x';
    }
}

/*--------------------------------------------------------------------------*/
/*! Map character to its corresponding token type
 *
 *  @return corresponding token type
 *  @return -1 if the character does not map to a valid token
 *
 *  @param c (in) character to convert
 */
/*--------------------------------------------------------------------------*/
static enum tokentype calceph_time_map_char_to_token(char c)
{
    switch (c)
    {
        case 'Q':
            return TOKEN_QUOTE;
        case '[':
            return TOKEN_LPAREN;
        case ']':
            return TOKEN_RPAREN;
        case ',':
            return TOKEN_COMMA;
        case '-':
            return TOKEN_DASH;
        case '.':
            return TOKEN_DOT;
        case '/':
            return TOKEN_SLASH;
        case ':':
            return TOKEN_COLON;
        case 'N':
            return TOKEN_AMPM;
        case 'O':
            return TOKEN_UTC_PLUS;
        case 'Z':
            return TOKEN_TIMEZONE;
        case 'b':
            return TOKEN_WHITESPACE;
        case 'd':
            return TOKEN_DOY_MARKER;
        case 'e':
            return TOKEN_ERA;
        case 'j':
            return TOKEN_JULIAN;
        case 'm':
            return TOKEN_MONTH;
        case 'o':
            return TOKEN_UTC_MINUS;
        case 's':
            return TOKEN_TIMESCALE;
        case 't':
            return TOKEN_T_SEPARATOR;
        case 'w':
            return TOKEN_WEEKDAY;
        case 'i':
            return TOKEN_INTEGER;
        case 'z':
            return TOKEN_Z;
        case '+':
            return TOKEN_PLUS;
        case 'n':
            return TOKEN_NUMBER;
        case 'Y':
            return TOKEN_YEAR;
        case 'D':
            return TOKEN_DAY_OF_MONTH;
        case 'y':
            return TOKEN_DAY_OF_YEAR;
        case 'H':
            return TOKEN_HOUR;
        case 'M':
            return TOKEN_MINUTE;
        case 'S':
            return TOKEN_SECOND;
        default:
            return -1;
    }
}

/*--------------------------------------------------------------------------*/
/*! Convert lexer into string representation
 *
 *  @param lexer (in)  lexer object
 *  @param res   (out) buffer to store the generated string
 */
/*--------------------------------------------------------------------------*/
void calceph_lexer_to_string(struct lexer *lexer, char *res)
{
    if (lexer == NULL)
        return;

    struct token *t;

    for (t = lexer->head; t != NULL; t = t->next)
    {
        *res++ = calceph_time_map_token_to_char(t->type);
    }
    *res = '\0';
}

/*--------------------------------------------------------------------------*/
/*! Match a keyword in the input string at the current position.
 *
 *  This function compares the substring starting at 'curpos' in 'timestr'
 *  with a list of known keywords. The comparison mode determines how strict
 *  the match must be:
 *    - mode 1 = exact match of the whole keyword
 *    - mode 2 = partial match of at least 3 characters (progressively extended)
 *
 *  @return length of the matched keyword (0 if no match)
 *
 *  @param timestr  (in)  input string to analyze
 *  @param curpos   (in)  current index in the string
 *  @param keywords (in)  array of keyword strings to match
 *  @param n        (in)  number of keywords in the array
 *  @param mode     (in)  matching mode (1 = exact, 2 = partial)
 */
/*--------------------------------------------------------------------------*/
static int match_keywords(const char *timestr, int curpos, const char *const *keywords, int n, int mode)
{
    int j;

    int len = strlen(timestr);

    int keywordlen;

    int res = 0;

    t_calcephcharvalue cptimestr;

    strcpy(cptimestr, timestr);

    calceph_strtoupper(cptimestr);

    for (j = 0; j < n && res == 0; j++)
    {
        keywordlen = strlen(keywords[j]);
        if (mode == 1)
        {
            if (curpos + keywordlen <= len && strncmp(cptimestr + curpos, keywords[j], keywordlen) == 0)
                res = keywordlen;
        }
        else if (mode == 2 && strncmp(cptimestr + curpos, keywords[j], 3) == 0)
        {
            res = 3;
            while (curpos + res < len && res < keywordlen && strncmp(cptimestr + curpos, keywords[j], res + 1) == 0)
                res++;
        }
    }
    return res;
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is a weekday
 *
 *  @return length of matched weekday string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int isweekday(const char *timestr, int curpos)
{
    static const char *weekdays[] = {
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY",
        "FRIDAY", "SATURDAY", "SUNDAY"
    };
    return match_keywords(timestr, curpos, weekdays, 7, 2);
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is a month
 *
 *  @return length of matched month string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int ismonth(const char *timestr, int curpos)
{
    static const char *months[] = {
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
        "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
    };
    return match_keywords(timestr, curpos, months, 12, 2);
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is a timezone
 *
 *  @return length of matched timezone string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int istimezone(const char *timestr, int curpos)
{
    static const char *zones[] = {
        "EST", "CST", "MST", "PST", "EDT", "CDT", "MDT", "PDT"
    };
    return match_keywords(timestr, curpos, zones, 8, 1);
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is a time system specifier
 *
 *  @return length of matched time system string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int istimesystem(const char *timestr, int curpos)
{
    static const char *timesys[] = { "TT", "TDT", "TDB", "UTC", "TAI", "TCB" };
    return match_keywords(timestr, curpos, timesys, 6, 1);
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is an era specifier
 *  
 *  @return length of matched era string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int iseraspecifier(const char *timestr, int curpos)
{
    static const char *eras[] = { "A.D.", "B.C.", "AD", "BC" };
    return match_keywords(timestr, curpos, eras, 4, 1);
}

/*--------------------------------------------------------------------------*/
/*! Check if substring at current position is an AM/PM specifier
 *
 *  @return length of matched AM/PM string (0 if no match)
 *
 *  @param timestr (in)  input string
 *  @param curpos  (in)  current index to check in the string
 */
/*--------------------------------------------------------------------------*/
static int isampmspecifier(const char *timestr, int curpos)
{
    static const char *ampm[] = { "A.M.", "P.M.", "AM", "PM" };
    return match_keywords(timestr, curpos, ampm, 4, 1);
}

/*-----------------------------------------------------------------*/
/* principal lexing function */
/*-----------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/
/*! Tokenize a time string into a lexer
 *
 *  @return pointer to the lexer containing the tokens
 *  @return NULL on error (invalid input, allocation failure, or unrecognized character)
 *
 *  @param timestr (in) input time string to tokenize
 */
/*--------------------------------------------------------------------------*/
struct lexer *calceph_time_tokenize(struct calceph_locale *pclocale, const char *timestr)
{
    if (timestr == NULL)
        return NULL;

    struct lexer *lexer = calceph_lexer_init(pclocale);

    if (lexer == NULL)
        return NULL;

    struct simple_token_map simple_tokens[] = {
        {'.', TOKEN_DOT},
        {'-', TOKEN_DASH},
        {'/', TOKEN_SLASH},
        {':', TOKEN_COLON},
        {'(', TOKEN_LPAREN},
        {')', TOKEN_RPAREN},
        {'\'', TOKEN_QUOTE},
        {',', TOKEN_COMMA},
        {'T', TOKEN_T_SEPARATOR},
        {'Z', TOKEN_Z},
        {'+', TOKEN_PLUS},
    };

    int curpos = 0;

    int endpos;

    int ret;

    int len = strlen(timestr);

    struct token *curtok;

    while (curpos < len)
    {
        if (timestr[curpos] >= '0' && timestr[curpos] <= '9')
        {
            endpos = curpos;

            while (endpos + 1 < len && timestr[endpos + 1] >= '0' && timestr[endpos + 1] <= '9')
                endpos++;

            curtok = calceph_token_init(TOKEN_INTEGER, curpos, endpos);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos = endpos + 1;
        }
        else if (timestr[curpos] == ' ' || timestr[curpos] == '\t')
        {
            endpos = curpos;

            while (endpos + 1 < len && (timestr[endpos + 1] == ' ' || timestr[endpos + 1] == '\t'))
                endpos++;

            curtok = calceph_token_init(TOKEN_WHITESPACE, curpos, endpos);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos = endpos + 1;
        }
        else if ((ret = isweekday(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_WEEKDAY, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if ((ret = ismonth(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_MONTH, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if ((ret = istimezone(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_TIMEZONE, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if (curpos + 3 < len && strncmp("UTC+", timestr + curpos, 4) == 0)
        {
            curtok = calceph_token_init(TOKEN_UTC_PLUS, curpos, curpos + 3);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += 4;
        }
        else if (curpos + 3 < len && strncmp("UTC-", timestr + curpos, 4) == 0)
        {
            curtok = calceph_token_init(TOKEN_UTC_MINUS, curpos, curpos + 3);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += 4;
        }
        else if ((ret = istimesystem(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_TIMESCALE, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if ((ret = iseraspecifier(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_ERA, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if ((ret = isampmspecifier(timestr, curpos)) != 0)
        {
            curtok = calceph_token_init(TOKEN_AMPM, curpos, curpos + ret - 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += ret;
        }
        else if (curpos + 1 < len && (strncmp("JD", timestr + curpos, 2) == 0 ||
                                      strncmp("jd", timestr + curpos, 2) == 0))
        {
            curtok = calceph_token_init(TOKEN_JULIAN, curpos, curpos + 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += 2;
        }
        else if (curpos + 1 < len && (strncmp("::", timestr + curpos, 2) == 0 ||
                                      strncmp("//", timestr + curpos, 2) == 0))
        {
            curtok = calceph_token_init(TOKEN_DOY_MARKER, curpos, curpos + 1);
            if (calceph_lexer_add_token(lexer, curtok) == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
            curpos += 2;
        }
        else
        {
            int matched = 0;

            size_t j;

            for (j = 0; j < sizeof(simple_tokens) / sizeof(simple_tokens[0]) && matched == 0; j++)
            {
                if (timestr[curpos] == simple_tokens[j].c)
                {
                    curtok = calceph_token_init(simple_tokens[j].type, curpos, curpos);
                    if (curtok == NULL || calceph_lexer_add_token(lexer, curtok) == 0)
                    {
                        calceph_lexer_destroy(lexer);
                        return NULL;
                    }
                    curpos++;
                    matched = 1;
                }
            }

            if (matched == 0)
            {
                calceph_lexer_destroy(lexer);
                return NULL;
            }
        }
    }

    return lexer;
}
