/*-----------------------------------------------------------------*/
/*!
  \file calcephtimelexer.h
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

/*-----------------------------------------------------------------*/
/* ENUM */
/*-----------------------------------------------------------------*/

/*! Char value in string representation of lexer : */
enum tokentype
{                              
    TOKEN_QUOTE,                /*!< Q */
    TOKEN_LPAREN,               /*!< [ */
    TOKEN_RPAREN,               /*!< ] */
    TOKEN_COMMA,                /*!< , */
    TOKEN_DASH,                 /*!< - */
    TOKEN_DOT,                  /*!< . */
    TOKEN_SLASH,                /*!< / */
    TOKEN_COLON,                /*!< : */
    TOKEN_AMPM,                 /*!< N */
    TOKEN_UTC_PLUS,             /*!< O */
    TOKEN_TIMEZONE,             /*!< Z */
    TOKEN_WHITESPACE,           /*!< b */
    TOKEN_DOY_MARKER,           /*!< d */
    TOKEN_ERA,                  /*!< e */
    TOKEN_JULIAN,               /*!< j */
    TOKEN_MONTH,                /*!< m */
    TOKEN_UTC_MINUS,            /*!< o */
    TOKEN_TIMESCALE,            /*!< s */
    TOKEN_T_SEPARATOR,          /*!< t */
    TOKEN_WEEKDAY,              /*!< w */
    TOKEN_INTEGER,              /*!< i */
    TOKEN_IGNORE,               /*!< x */
    TOKEN_Z,                    /*!< z */
    TOKEN_PLUS,                 /*!< + */
    TOKEN_NUMBER,               /*!< n */
    TOKEN_YEAR,                 /*!< Y */
    TOKEN_DAY_OF_MONTH,         /*!< D */
    TOKEN_DAY_OF_YEAR,          /*!< y */
    TOKEN_HOUR,                 /*!< H */
    TOKEN_MINUTE,               /*!< M */
    TOKEN_SECOND                /*!< S */
};

/*-----------------------------------------------------------------*/
/* STRUCT */
/*-----------------------------------------------------------------*/

/*! Represents a single token in the lexer */
struct token
{
    enum tokentype type;
    int start;
    int end;
    struct token *next;
    struct token *prec;
};

/*! Represents a lexer containing a list of tokens */
struct lexer
{
    struct calceph_locale *ephlocale;
    struct token *head;
    struct token *tail;
    int ntokens;
};

/*-----------------------------------------------------------------*/
/* FUNCTIONS */
/*-----------------------------------------------------------------*/

/*! Allocates and initializes a new token */
struct token *calceph_token_init(enum tokentype type, int start, int end);

/*! Allocates a copy of the given token */
struct token *calceph_token_copy(struct token *token);

/*! Reads the substring corresponding to a token */
void calceph_token_read(struct token *token, const char *timestr, t_calcephcharvalue res);

/*! Allocates and initializes an empty lexer */
struct lexer *calceph_lexer_init(struct calceph_locale *pclocale);

/*! Allocates a copy of the given lexer and its tokens */
struct lexer *calceph_lexer_copy(struct lexer *lexer);

/*! Frees all tokens in a lexer without freeing the lexer itself */
void calceph_lexer_clear(struct lexer *lexer);

/*! Frees all tokens and the lexer structure itself */
void calceph_lexer_destroy(struct lexer *lexer);

/*! Replaces the contents of one lexer with another */
void calceph_lexer_replace(struct lexer *lexer, struct lexer *rlexer);

/*! Adds a token to the end of a lexer */
int calceph_lexer_add_token(struct lexer *lexer, struct token *token);

/*! Removes a specific token from a lexer */
int calceph_lexer_remove_token(struct lexer *lexer, struct token *token);

/*! Returns the token at the specified index */
struct token *calceph_lexer_get_token_at_index(struct lexer *lexer, int index);

/*! Replaces tokens in a lexer based on a pattern and replacement rule */
int calceph_lexer_replace_tokens_from_pattern(struct lexer *lexer, const char *pattern, const char *rpattern, int mode);

/*! Converts the entire lexer into its string representation */
void calceph_lexer_to_string(struct lexer *lexer, char *res);

/*! Tokenizes a date/time string into a lexer structure */
struct lexer *calceph_time_tokenize(struct calceph_locale *pclocale, const char *timestr);
