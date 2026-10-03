/*-----------------------------------------------------------------*/
/*!
  \file interfacemex.c
  \brief MEX interface for the matlab/octave software

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright,  2018-2026, CNRS
   email of the author : Mickael.Gastineau@obspm.fr

  History:
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
#include "calceph.h"
#include "mex.h"
#include <stdio.h>
#include <string.h>

/*--------------------------------------------------------------------------*/
/*! enumeration of possible actions.
 */
/*--------------------------------------------------------------------------*/
enum emexaction
{
    ACTION_OPEN,                /*!< open */
    ACTION_CLOSE,               /*!< close */
    ACTION_PREFETCH,            /*!< prefetch */
    ACTION_ISTHREADSAFE,        /*!< isthreadsafe */
    ACTION_COMPUTE,             /*!< compute */
    ACTION_COMPUTE_UNIT,        /*!< compute_unit */
    ACTION_COMPUTE_ORDER,       /*!< compute_order */
    ACTION_ORIENT_UNIT,         /*!< orient_unit */
    ACTION_ORIENT_ORDER,        /*!< orient_order */
    ACTION_ROTANGMOM_UNIT,      /*!< rotangmom_unit */
    ACTION_ROTANGMOM_ORDER,     /*!< rotangmom_order */
    ACTION_GETCONSTANT,         /*!< getconstant */
    ACTION_GETCONSTANTSD,       /*!< getconstantsd */
    ACTION_GETCONSTANTVD,       /*!< getconstantvd */
    ACTION_GETCONSTANTSS,       /*!< getconstantss */
    ACTION_GETCONSTANTVS,       /*!< getconstantvs */
    ACTION_GETCONSTANTCOUNT,    /*!< getconstantcount */
    ACTION_GETCONSTANTINDEX,    /*!< getconstantindex */
    ACTION_GETFOV,              /*!< getfov */
    ACTION_GETIDBYNAME,         /*!< getidbyname */
    ACTION_GETNAMEBYIDSS,       /*!< getnamebyidss */
    ACTION_GETTIMESCALE,        /*!< gettimescale */
    ACTION_GETTIMESPAN,         /*!< gettimespan */
    ACTION_GETPOSITIONRECORDCOUNT,  /*!< getpositionrecordcount */
    ACTION_GETPOSITIONRECORDINDEX,  /*!< getpositionrecordindex */
    ACTION_GETPOSITIONRECORDINDEX2, /*!< getpositionrecordindex2 */
    ACTION_GETORIENTRECORDCOUNT,    /*!< getorientrecordcount */
    ACTION_GETORIENTRECORDINDEX,    /*!< getorientrecordindex */
    ACTION_GETORIENTRECORDINDEX2,   /*!< getorientrecordindex2 */
    ACTION_GETFILEVERSION,      /*!< getfileversion */
    ACTION_SETERRORHANDLER,     /*!< seterrorhandler */
    ACTION_GETVERSION_STR,      /*!< getversion_str */
    ACTION_GETMAXSUPPORTEDORDER,    /*!< getmaxsupportedorder */
    ACTION_TIME_JD_TO_CAL,      /*! time_jd_to_cal */
    ACTION_TIME_CAL_TO_JD,      /*! time_cal_to_jd */
    ACTION_TIME_STR_TO_JD,      /*! time_str_to_jd */
    ACTION_SET_RELATION_SHIP_TT_TDB,    /*! time_set_relationship_tt_tdb */
    ACTION_TIME_JD_TDB_TO_JD_TT,    /*! time_jd_tdb_to_jd_tt */
    ACTION_TIME_JD_TT_TO_JD_TDB,    /*! time_jd_tt_to_jd_tdb */
    ACTION_TIME_JD_TDB_TO_JD_TCB,   /*! time_jd_tdb_to_jd_tcb */
    ACTION_TIME_JD_TCB_TO_JD_TDB,   /*! time_jd_tcb_to_jd_tdb */
    ACTION_TIME_CAL_UTC_TO_JD_TDB,  /*! time_cal_utc_to_jd_tdb */
    ACTION_TIME_JD_TDB_TO_CAL_UTC,  /*! time_jd_tdb_to_cal_utc */
    ACTION_TIME_STR_UTC_TO_JD_TDB,  /*! time_str_utc_to_jd_tdb */
    ACTION_TIME_STR_ANY_TO_JD_TDB,  /*! time_str_any_to_jd_tdb */
    ACTION_TIME_SPACECRAFT_CLOCK_TO_JD_TDB, /*! time_str_spacecraft_clock_to_jd_tdb */
    ACTION_TIME_JD_TDB_TO_SPACECRAFT_CLOCK, /*! time_jd_tdb_to_str_spacecraft_clock */
    ACTION_INVALID              /*!< invalid */
};

/*--------------------------------------------------------------------------*/
/*! equivalence structure to store pointer as big integer.
 */
/*--------------------------------------------------------------------------*/
union upointer
{
    long long ptr_as_integer;
    void *ptr_as_pointer;
};

/*--------------------------------------------------------------------------*/
/*! Mex user function for error handler.
 */
/*--------------------------------------------------------------------------*/
static char *s_calceph_mex_erroruserfuncname = NULL;

/*--------------------------------------------------------------------------*/
/*! return the enumeration corresponding to the specified action.
   return ACTION_INVALID on error.

  @param caction (in) requested action
*/
/*--------------------------------------------------------------------------*/
static enum emexaction findaction(const char *caction)
{
    enum emexaction eres = ACTION_INVALID;

    if (strcmp(caction, "open") == 0)
        eres = ACTION_OPEN;
    else if (strcmp(caction, "close") == 0)
        eres = ACTION_CLOSE;
    else if (strcmp(caction, "prefetch") == 0)
        eres = ACTION_PREFETCH;
    else if (strcmp(caction, "isthreadsafe") == 0)
        eres = ACTION_ISTHREADSAFE;
    else if (strcmp(caction, "compute") == 0)
        eres = ACTION_COMPUTE;
    else if (strcmp(caction, "compute_unit") == 0)
        eres = ACTION_COMPUTE_UNIT;
    else if (strcmp(caction, "compute_order") == 0)
        eres = ACTION_COMPUTE_ORDER;
    else if (strcmp(caction, "orient_unit") == 0)
        eres = ACTION_ORIENT_UNIT;
    else if (strcmp(caction, "orient_order") == 0)
        eres = ACTION_ORIENT_ORDER;
    else if (strcmp(caction, "rotangmom_unit") == 0)
        eres = ACTION_ROTANGMOM_UNIT;
    else if (strcmp(caction, "rotangmom_order") == 0)
        eres = ACTION_ROTANGMOM_ORDER;
    else if (strcmp(caction, "getconstant") == 0)
        eres = ACTION_GETCONSTANT;
    else if (strcmp(caction, "getconstantsd") == 0)
        eres = ACTION_GETCONSTANTSD;
    else if (strcmp(caction, "getconstantvd") == 0)
        eres = ACTION_GETCONSTANTVD;
    else if (strcmp(caction, "getconstantss") == 0)
        eres = ACTION_GETCONSTANTSS;
    else if (strcmp(caction, "getconstantvs") == 0)
        eres = ACTION_GETCONSTANTVS;
    else if (strcmp(caction, "getconstantcount") == 0)
        eres = ACTION_GETCONSTANTCOUNT;
    else if (strcmp(caction, "getconstantindex") == 0)
        eres = ACTION_GETCONSTANTINDEX;
    else if (strcmp(caction, "getidbyname") == 0)
        eres = ACTION_GETIDBYNAME;
    else if (strcmp(caction, "getnamebyidss") == 0)
        eres = ACTION_GETNAMEBYIDSS;
    else if (strcmp(caction, "getpositionrecordcount") == 0)
        eres = ACTION_GETPOSITIONRECORDCOUNT;
    else if (strcmp(caction, "getpositionrecordindex") == 0)
        eres = ACTION_GETPOSITIONRECORDINDEX;
    else if (strcmp(caction, "getpositionrecordindex2") == 0)
        eres = ACTION_GETPOSITIONRECORDINDEX2;
    else if (strcmp(caction, "getorientrecordcount") == 0)
        eres = ACTION_GETORIENTRECORDCOUNT;
    else if (strcmp(caction, "getorientrecordindex") == 0)
        eres = ACTION_GETORIENTRECORDINDEX;
    else if (strcmp(caction, "getorientrecordindex2") == 0)
        eres = ACTION_GETORIENTRECORDINDEX2;
    else if (strcmp(caction, "gettimescale") == 0)
        eres = ACTION_GETTIMESCALE;
    else if (strcmp(caction, "gettimespan") == 0)
        eres = ACTION_GETTIMESPAN;
    else if (strcmp(caction, "getfileversion") == 0)
        eres = ACTION_GETFILEVERSION;
    else if (strcmp(caction, "seterrorhandler") == 0)
        eres = ACTION_SETERRORHANDLER;
    else if (strcmp(caction, "getversion_str") == 0)
        eres = ACTION_GETVERSION_STR;
    else if (strcmp(caction, "getmaxsupportedorder") == 0)
        eres = ACTION_GETMAXSUPPORTEDORDER;
    else if (strcmp(caction, "time_jd_to_cal") == 0)
        eres = ACTION_TIME_JD_TO_CAL;
    else if (strcmp(caction, "time_cal_to_jd") == 0)
        eres = ACTION_TIME_CAL_TO_JD;
    else if (strcmp(caction, "time_str_to_jd") == 0)
        eres = ACTION_TIME_STR_TO_JD;
    else if (strcmp(caction, "time_set_relationship_tt_tdb") == 0)
        eres = ACTION_SET_RELATION_SHIP_TT_TDB;
    else if (strcmp(caction, "time_jd_tdb_to_jd_tt") == 0)
        eres = ACTION_TIME_JD_TDB_TO_JD_TT;
    else if (strcmp(caction, "time_jd_tt_to_jd_tdb") == 0)
        eres = ACTION_TIME_JD_TT_TO_JD_TDB;
    else if (strcmp(caction, "time_jd_tdb_to_jd_tcb") == 0)
        eres = ACTION_TIME_JD_TDB_TO_JD_TCB;
    else if (strcmp(caction, "time_jd_tcb_to_jd_tdb") == 0)
        eres = ACTION_TIME_JD_TCB_TO_JD_TDB;
    else if (strcmp(caction, "time_cal_utc_to_jd_tdb") == 0)
        eres = ACTION_TIME_CAL_UTC_TO_JD_TDB;
    else if (strcmp(caction, "time_jd_tdb_to_cal_utc") == 0)
        eres = ACTION_TIME_JD_TDB_TO_CAL_UTC;
    else if (strcmp(caction, "time_str_utc_to_jd_tdb") == 0)
        eres = ACTION_TIME_STR_UTC_TO_JD_TDB;
    else if (strcmp(caction, "time_str_any_to_jd_tdb") == 0)
        eres = ACTION_TIME_STR_ANY_TO_JD_TDB;
    else if (strcmp(caction, "time_str_spacecraft_clock_to_jd_tdb") == 0)
        eres = ACTION_TIME_SPACECRAFT_CLOCK_TO_JD_TDB;
    else if (strcmp(caction, "time_jd_tdb_to_str_spacecraft_clock") == 0)
        eres = ACTION_TIME_JD_TDB_TO_SPACECRAFT_CLOCK;
    else if (strcmp(caction, "getfov") == 0)
        eres = ACTION_GETFOV;
    return eres;
}

/*--------------------------------------------------------------------------*/
/*! check if value is a scalar. Octave (version<4.2) does not have mxIsScalar.

  @param value (in) mex data
*/
/*--------------------------------------------------------------------------*/
static bool calceph_mxIsScalar(const mxArray *value)
{
    mxClassID id = mxGetClassID(value);

    return (id == mxDOUBLE_CLASS || id == mxSINGLE_CLASS || id == mxINT8_CLASS || id == mxUINT8_CLASS
            || id == mxINT16_CLASS || id == mxUINT16_CLASS || id == mxINT32_CLASS || id == mxUINT32_CLASS
            || id == mxINT64_CLASS || id == mxUINT64_CLASS);
}

/*--------------------------------------------------------------------------*/
/*! check that the expected and provided number of arguments are equal.

  @param nlhs (in) provided left argument
  @param nlhs_expected (in) expected left argument
  @param nrhs (in) provided right argument
  @param nrhs_expected (in) expected right argument
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_checknbargs(int nlhs, int nrhs, int nlhs_expected, int nrhs_expected)
{
    if (nlhs != nlhs_expected)
        mexErrMsgTxt("Wrong number of left assignment ");
    if (nrhs != nrhs_expected)
        mexErrMsgTxt("Wrong number of argument");
}

/*--------------------------------------------------------------------------*/
/*! check if value is an array of double.
  @param value (in) mex data
*/
/*--------------------------------------------------------------------------*/
static bool calceph_mxisarray(const mxArray *value)
{
    return (mxGetClassID(value) == mxDOUBLE_CLASS && mxGetNumberOfElements(value) > 1);
}

/*--------------------------------------------------------------------------*/
/*! return 1 if the two values are arrays of floating-point numbers
    and they have of same length >1.
    otherwise return 0

  @param value1 (in) first argument to check
  @param value2 (in) second argument to check
*/
/*--------------------------------------------------------------------------*/
static int calceph_mex_isarray_and_samesize(const mxArray *value1, const mxArray *value2)
{
    int res = 0;

    if (calceph_mxisarray(value1) && calceph_mxisarray(value2))
    {
        res = (mxGetNumberOfElements(value1) == mxGetNumberOfElements(value2)) ? 1 : 0;
    }
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the pointer to a big integer for the mex format.

  @param phandle (in) ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static mxArray *calceph_convert_handle_to_integer(t_calcephbin *phandle)
{
    mxArray *res = NULL;

    union upointer auvp;

    long long *ip;

    res = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
    ip = (long long *) mxGetData(res);
    auvp.ptr_as_integer = 0;
    auvp.ptr_as_pointer = phandle;
    *ip = auvp.ptr_as_integer;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the a big integer for the mex format
   to the pointer of the ephemeris descriptor .

  @param phandle (in) ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static t_calcephbin *calceph_convert_integer_to_handle(const mxArray *ahandle)
{
    t_calcephbin *res = NULL;

    int m, n;

    union upointer auvp;

    const long long *ip;

    m = mxGetM(ahandle);
    n = mxGetN(ahandle);
    if (m != 1 || n != 1)
        mexErrMsgTxt("Not a valod ephemeris descriptor");

    ip = (const long long *) mxGetData(ahandle);
    auvp.ptr_as_integer = *ip;
    res = auvp.ptr_as_pointer;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the C integer to a mex integer.

  @param value (in) integer
*/
/*--------------------------------------------------------------------------*/
static mxArray *calceph_convert_integer_to_mex(int value)
{
    int *ip;

    mxArray *res;

    res = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    ip = (int *) mxGetData(res);
    *ip = value;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the mex integer to a C integer.

  @param value (in) integer
*/
/*--------------------------------------------------------------------------*/
static int calceph_convert_mex_to_integer(const mxArray *value)
{
    int res;

    if (!calceph_mxIsScalar(value))
        mexErrMsgTxt("Wrong type for an argument: must be an integer scalar");

    res = (int) mxGetScalar(value);
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the C double to a mex double.

  @param value (in) double
*/
/*--------------------------------------------------------------------------*/
static mxArray *calceph_convert_double_to_mex(double value)
{
    double *ip;

    mxArray *res;

    res = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
    ip = (double *) mxGetData(res);
    *ip = value;
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the mex double to a C double.

  @param value (in) double
*/
/*--------------------------------------------------------------------------*/
static double calceph_convert_mex_to_double(const mxArray *value)
{
    if (!calceph_mxIsScalar(value))
        mexErrMsgTxt("Wrong type for an argument: must be a double scalar");

    return mxGetScalar(value);
}

/*--------------------------------------------------------------------------*/
/*! convert the array of C double to a mex matrix of double.

  @param value (in) double
*/
/*--------------------------------------------------------------------------*/
static mxArray *calceph_convert_array_double_to_mex(const double *array, int n)
{
    int k;

    double *ip;

    mxArray *res;

    res = mxCreateNumericMatrix(1, n, mxDOUBLE_CLASS, mxREAL);
    ip = (double *) mxGetData(res);
    for (k = 0; k < n; k++)
        ip[k] = array[k];
    return res;
}

/*--------------------------------------------------------------------------*/
/*! convert the mex matrix of double to an array of C double.

  @param value (in) double
*/
/*--------------------------------------------------------------------------*/
static const double *calceph_convert_mex_to_array_double(const mxArray *array)
{
    return mxGetPr(array);
}

/*--------------------------------------------------------------------------*/
/*! convert the C string to a mex string.

  @param value (in) C string
*/
/*--------------------------------------------------------------------------*/
static mxArray *calceph_convert_cstring_to_mex(const char *value)
{
    return mxCreateString(value);
}

/*--------------------------------------------------------------------------*/
/*! convert the mex string to C string.
    The returned C string must be freed with mxFree.

  @param value (in) Mex string
*/
/*--------------------------------------------------------------------------*/
static char *calceph_convert_mex_to_cstring(const mxArray *value)
{
    char *res = mxArrayToString(value);

    if (res == NULL)
        mexErrMsgTxt("Wrong type for an argument: must be an string of characters");
    return res;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_open.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments : plhs[0] = ephemeris descriptor
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = string
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_open_onefile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *cfilename;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    cfilename = mxArrayToString(prhs[1]);
    if (cfilename == NULL)
        mexErrMsgTxt("First argument of open must a string");
    phandle = calceph_open(cfilename);
    mxFree(cfilename);
    plhs[0] = calceph_convert_handle_to_integer(phandle);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_open_array.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments : plhs[0] = ephemeris descriptor
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = string
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_open_cellstrfiles(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    t_calcephbin *phandle;

    char **arraystr;

    mxArray *cell;

    int j, mrows, ncols, ntot;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);

    mrows = mxGetM(prhs[1]);
    ncols = mxGetN(prhs[1]);
    ntot = mrows * ncols;
    arraystr = (char **) mxMalloc(ntot * sizeof(char *));
    if (arraystr == NULL)
        mexErrMsgTxt("Not enough memory");

    for (j = 0; j < ntot; j++)
    {
        cell = mxGetCell(prhs[1], j);
        if (cell == NULL)
            mexErrMsgTxt("First argument of open must a Cell of string (cellstr)");
        arraystr[j] = mxArrayToString(cell);
        if (arraystr[j] == NULL)
            mexErrMsgTxt("First argument of open must a string");
    }
    phandle = calceph_open_array(ntot, (const char *const *) arraystr);
    for (j = 0; j < ntot; j++)
    {
        mxFree(arraystr[j]);
    }
    mxFree(arraystr);
    plhs[0] = calceph_convert_handle_to_integer(phandle);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_open.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments : plhs[0] = ephemeris descriptor
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = string
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_open(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    if (!mxIsCell(prhs[1]))
        calceph_mex_open_onefile(nlhs, plhs, nrhs, prhs);
    else
        calceph_mex_open_cellstrfiles(nlhs, plhs, nrhs, prhs);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_close.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_close(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    t_calcephbin *phandle;

    (void) plhs;

    calceph_mex_checknbargs(nlhs, nrhs, 0, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        calceph_close(phandle);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_prefetch.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments  plhs[0] = return code
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_prefetch(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_prefetch(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_isthreadsafe.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments  plhs[0] = return code
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_isthreadsafe(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_isthreadsafe(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute on one value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, center;

    double PV[6];

    calceph_mex_checknbargs(nlhs, nrhs, 2, 6);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
        res = calceph_compute(phandle, jd0, dt, target, center, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 6);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute on one value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, center;

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    double PV[6];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 6, mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_compute(phandle, jd0[k], dt[k], target, center, PV);
            for (j = 0; j < 6; j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute on one value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 6);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_compute_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_compute_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_unit for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_unit_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, center, unit;

    double PV[6];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    unit = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
        res = calceph_compute_unit(phandle, jd0, dt, target, center, unit, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 6);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_unit for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_unit_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, center, unit;

    double PV[6];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    unit = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 6, mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_compute_unit(phandle, jd0[k], dt[k], target, center, unit, PV);
            for (j = 0; j < 6; j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_unit.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_unit(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 7);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_compute_unit_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_compute_unit_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_order for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
  prhs[7] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_order_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, center, unit, order;

    double PV[12];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    unit = calceph_convert_mex_to_integer(prhs[6]);
    order = calceph_convert_mex_to_integer(prhs[7]);
    if (phandle != NULL)
        res = calceph_compute_order(phandle, jd0, dt, target, center, unit, order, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 3 * (order + 1));
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_order for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
  prhs[7] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_order_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, center, unit, order;

    double PV[12];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    center = calceph_convert_mex_to_integer(prhs[5]);
    unit = calceph_convert_mex_to_integer(prhs[6]);
    order = calceph_convert_mex_to_integer(prhs[7]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 3 * (order + 1), mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_compute_order(phandle, jd0[k], dt[k], target, center, unit, order, PV);
            for (j = 0; j < 3 * (order + 1); j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_compute_order.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = center
  prhs[6] = unit
  prhs[7] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_compute_order(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 8);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_compute_order_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_compute_order_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_unit for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_unit_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, unit;

    double PV[6];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
        res = calceph_orient_unit(phandle, jd0, dt, target, unit, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 6);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_unit for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_unit_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, unit;

    double PV[6];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 6, mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_orient_unit(phandle, jd0[k], dt[k], target, unit, PV);
            for (j = 0; j < 6; j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_unit.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_unit(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 6);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_orient_unit_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_orient_unit_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_order for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_order_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, unit, order;

    double PV[12];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    order = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
        res = calceph_orient_order(phandle, jd0, dt, target, unit, order, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 3 * (order + 1));
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_order for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_order_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, unit, order;

    double PV[12];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    order = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 3 * (order + 1), mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_orient_order(phandle, jd0[k], dt[k], target, unit, order, PV);
            for (j = 0; j < 3 * (order + 1); j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_orient_order.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_orient_order(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 7);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_orient_order_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_orient_order_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_unit for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_unit_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, unit;

    double PV[6];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
        res = calceph_rotangmom_unit(phandle, jd0, dt, target, unit, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 6);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_unit for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_unit_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, unit;

    double PV[6];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 6, mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_rotangmom_unit(phandle, jd0[k], dt[k], target, unit, PV);
            for (j = 0; j < 6; j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_unit.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_unit(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 6);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_rotangmom_unit_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_rotangmom_unit_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_order for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_order_scalar(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0, dt;

    int target, unit, order;

    double PV[12];

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_double(prhs[2]);
    dt = calceph_convert_mex_to_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    order = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
        res = calceph_rotangmom_order(phandle, jd0, dt, target, unit, order, PV);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_array_double_to_mex(PV, 3 * (order + 1));
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_order for an array of values.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_order_array(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    const double *jd0;

    const double *dt;

    int target, unit, order;

    double PV[12];

    mxArray *array = NULL;

    int k, n, j;

    double *data;

    (void) nlhs;
    (void) nrhs;

    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0 = calceph_convert_mex_to_array_double(prhs[2]);
    dt = calceph_convert_mex_to_array_double(prhs[3]);
    target = calceph_convert_mex_to_integer(prhs[4]);
    unit = calceph_convert_mex_to_integer(prhs[5]);
    order = calceph_convert_mex_to_integer(prhs[6]);
    if (phandle != NULL)
    {
        n = mxGetNumberOfElements(prhs[2]);
        array = mxCreateNumericMatrix(n, 3 * (order + 1), mxDOUBLE_CLASS, mxREAL);
        data = mxGetPr(array);
        res = 1;
        for (k = 0; k < n && res == 1; k++)
        {
            res = calceph_rotangmom_order(phandle, jd0[k], dt[k], target, unit, order, PV);
            for (j = 0; j < 3 * (order + 1); j++)
                data[j * n + k] = PV[j];
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = array;
}

/*--------------------------------------------------------------------------*/
/*! call calceph_rotangmom_order.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = Position/Velocity/Acceleration/Jerks
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = JD0
  prhs[3] = dt
  prhs[4] = target
  prhs[5] = unit
  prhs[6] = order
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_rotangmom_order(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    calceph_mex_checknbargs(nlhs, nrhs, 2, 7);
    if (calceph_mex_isarray_and_samesize(prhs[2], prhs[3]))
    {
        calceph_mex_rotangmom_order_array(nlhs, plhs, nrhs, prhs);
    }
    else
    {
        calceph_mex_rotangmom_order_scalar(nlhs, plhs, nrhs, prhs);
    }
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstant.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name of the constant
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstant(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double value = 0.E0;

    char *cname;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_getconstant(phandle, cname, &value);
    mxFree(cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(value);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantsd.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name of the constant
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantsd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double value = 0.E0;

    char *cname;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_getconstantsd(phandle, cname, &value);
    mxFree(cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(value);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantvd.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = array of value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name of the constant
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantvd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double value = 0.E0;

    char *cname;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_getconstantsd(phandle, cname, &value);
    if (res >= 1)
    {
        plhs[1] = mxCreateNumericMatrix(1, res, mxDOUBLE_CLASS, mxREAL);
        res = calceph_getconstantvd(phandle, cname, (double *) mxGetData(plhs[1]), res);
    }
    mxFree(cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantss.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name of the constant
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantss(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    t_calcephcharvalue value;

    char *cname;

    value[0] = '\0';
    calceph_mex_checknbargs(nlhs, nrhs, 2, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_getconstantss(phandle, cname, value);
    mxFree(cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_cstring_to_mex(value);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantvs.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = array of value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name of the constant
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantvs(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0, k;

    t_calcephbin *phandle;

    t_calcephcharvalue value;

    char *cname;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_getconstantss(phandle, cname, value);
    if (res >= 1)
    {
        t_calcephcharvalue *arvalue;

        mwSize dims[1];

        arvalue = (t_calcephcharvalue *) mxMalloc(sizeof(t_calcephcharvalue) * res);
        if (arvalue == NULL)
            mexErrMsgTxt("Not enough memory to store values");
        res = calceph_getconstantvs(phandle, cname, arvalue, res);
        dims[0] = res;
        plhs[1] = mxCreateCellMatrix(1, res);
        for (k = 0; k < res; k++)
            mxSetCell(plhs[1], k, calceph_convert_cstring_to_mex(arvalue[k]));

        mxFree(arvalue);
    }
    mxFree(cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantcount.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = number of constants
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantcount(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_getconstantcount(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getconstantindex.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = name of constant
  plhs[2] = value of constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = index
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getconstantindex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int index;

    double value = 0.E0;

    char cname[CALCEPH_MAX_CONSTANTNAME];

    calceph_mex_checknbargs(nlhs, nrhs, 3, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    index = calceph_convert_mex_to_integer(prhs[2]);
    cname[0] = '\0';
    if (phandle != NULL)
        res = calceph_getconstantindex(phandle, index, cname, &value);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_cstring_to_mex(cname);
    plhs[2] = calceph_convert_double_to_mex(value);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getpositionrecordcount.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = number of position records
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getpositionrecordcount(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_getpositionrecordcount(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getpositionrecordindex.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = target
  plhs[2] = center
  plhs[3] = firsttime
  plhs[4] = lasttime
  plhs[5] = frame
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = index
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getpositionrecordindex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int index;

    int target = 0, center = 0;

    double firsttime = 0.0, lasttime = 0.0;

    int frame = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 6, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    index = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
        res = calceph_getpositionrecordindex(phandle, index, &target, &center, &firsttime, &lasttime, &frame);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(target);
    plhs[2] = calceph_convert_integer_to_mex(center);
    plhs[3] = calceph_convert_double_to_mex(firsttime);
    plhs[4] = calceph_convert_double_to_mex(lasttime);
    plhs[5] = calceph_convert_integer_to_mex(frame);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getpositionrecordindex2.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = target
  plhs[2] = center
  plhs[3] = firsttime
  plhs[4] = lasttime
  plhs[5] = frame
  plhs[6] = segid
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = index
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getpositionrecordindex2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int index;

    int target = 0, center = 0;

    double firsttime = 0.0, lasttime = 0.0;

    int frame = 0;

    int segid = -1;

    calceph_mex_checknbargs(nlhs, nrhs, 7, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    index = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
        res = calceph_getpositionrecordindex2(phandle, index, &target, &center, &firsttime, &lasttime, &frame, &segid);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(target);
    plhs[2] = calceph_convert_integer_to_mex(center);
    plhs[3] = calceph_convert_double_to_mex(firsttime);
    plhs[4] = calceph_convert_double_to_mex(lasttime);
    plhs[5] = calceph_convert_integer_to_mex(frame);
    plhs[6] = calceph_convert_integer_to_mex(segid);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getorientrecordcount.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = number of position records
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getorientrecordcount(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_getorientrecordcount(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getorientrecordindex.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = target
  plhs[2] = firsttime
  plhs[3] = lasttime
  plhs[4] = frame
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = index
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getorientrecordindex(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int index;

    int target = 0;

    double firsttime = 0.0, lasttime = 0.0;

    int frame = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 5, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    index = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
        res = calceph_getorientrecordindex(phandle, index, &target, &firsttime, &lasttime, &frame);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(target);
    plhs[2] = calceph_convert_double_to_mex(firsttime);
    plhs[3] = calceph_convert_double_to_mex(lasttime);
    plhs[4] = calceph_convert_integer_to_mex(frame);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getorientrecordindex2.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = target
  plhs[2] = firsttime
  plhs[3] = lasttime
  plhs[4] = frame
  plhs[5] = segid
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = index
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getorientrecordindex2(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int index;

    int target = 0;

    double firsttime = 0.0, lasttime = 0.0;

    int frame = 0;

    int segid = -1;

    calceph_mex_checknbargs(nlhs, nrhs, 6, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    index = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
        res = calceph_getorientrecordindex2(phandle, index, &target, &firsttime, &lasttime, &frame, &segid);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(target);
    plhs[2] = calceph_convert_double_to_mex(firsttime);
    plhs[3] = calceph_convert_double_to_mex(lasttime);
    plhs[4] = calceph_convert_integer_to_mex(frame);
    plhs[5] = calceph_convert_integer_to_mex(segid);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_gettimescale.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = time scale
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_gettimescale(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_gettimescale(phandle);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_gettimespan.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = first time of the ephemeris
  plhs[2] = last time of the ephemeris
  plhs[3] = continuous
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_gettimespan(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double firsttime = 0.0, lasttime = 0.0;

    int continuous = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 4, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_gettimespan(phandle, &firsttime, &lasttime, &continuous);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(firsttime);
    plhs[2] = calceph_convert_double_to_mex(lasttime);
    plhs[3] = calceph_convert_integer_to_mex(continuous);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getidbyname

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = id
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = name
  prhs[3] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getidbyname(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int id = -1;

    int unit;

    char *cname;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    cname = calceph_convert_mex_to_cstring(prhs[2]);
    unit = calceph_convert_mex_to_integer(prhs[3]);
    if (phandle != NULL)
        res = calceph_getidbyname(phandle, cname, unit, &id);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(id);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getnamebyidss

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = name
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = id
  prhs[3] = unit
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getnamebyidss(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephcharvalue cname;

    t_calcephbin *phandle;

    int id = -1;

    int unit;

    calceph_mex_checknbargs(nlhs, nrhs, 2, 4);
    cname[0] = '\0';
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    id = calceph_convert_mex_to_integer(prhs[2]);
    unit = calceph_convert_mex_to_integer(prhs[3]);
    if (phandle != NULL)
        res = calceph_getnamebyidss(phandle, id, unit, cname);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_cstring_to_mex(cname);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getfileversion.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = value of the constant
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getfileversion(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    t_calcephcharvalue value;

    value[0] = '\0';
    calceph_mex_checknbargs(nlhs, nrhs, 2, 2);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    if (phandle != NULL)
        res = calceph_getfileversion(phandle, value);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_cstring_to_mex(value);
}

/*--------------------------------------------------------------------------*/
/*! MEX error handler for user func call.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  @param nrhs (in) right argument
  @param prhs (in) right arguments
  prhs[1] = type handler
  prhs[2] = usee function
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_errorhandler_userfunc(const char *msg)
{
    mxArray *prhs[1];

    prhs[0] = calceph_convert_cstring_to_mex(msg);
    mexCallMATLAB(0, NULL, 1, prhs, s_calceph_mex_erroruserfuncname);
    mxDestroyArray(prhs[0]);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_seterrorhandler.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  @param nrhs (in) right argument
  @param prhs (in) right arguments
  prhs[1] = type handler
  prhs[2] = usee function
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_seterrorhandler(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *p;

    int typehandler;

    (void) plhs;

    if (s_calceph_mex_erroruserfuncname != NULL)
    {
        free(s_calceph_mex_erroruserfuncname);
        s_calceph_mex_erroruserfuncname = NULL;
    }
    calceph_mex_checknbargs(nlhs, nrhs, 0, 3);
    typehandler = calceph_convert_mex_to_integer(prhs[1]);
    p = calceph_convert_mex_to_cstring(prhs[2]);
    s_calceph_mex_erroruserfuncname = (char *) malloc((strlen(p) + 1) * sizeof(char));
    if (s_calceph_mex_erroruserfuncname == NULL)
        mexErrMsgTxt("Not enough memory");
    strcpy(s_calceph_mex_erroruserfuncname, p);
    calceph_seterrorhandler(typehandler, calceph_mex_errorhandler_userfunc);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getversion_str.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments  plhs[0] = version string
  @param nrhs (in) right argument
  @param prhs (in) right arguments
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getversion_str(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char version[CALCEPH_MAX_CONSTANTNAME];

    (void) prhs;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 1);
    calceph_getversion_str(version);
    plhs[0] = calceph_convert_cstring_to_mex(version);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getmaxsupportedorder.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = supported order
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = segment id
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getmaxsupportedorder(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int idseg, res;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 2);
    idseg = calceph_convert_mex_to_integer(prhs[1]);
    res = calceph_getmaxsupportedorder(idseg);
    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_getfov for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = shape
  plhs[2] = frame
  plhs[3] = vector
  plhs[4] = arraybounds
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = instrumentid
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_getfov(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0, bounds;

    t_calcephbin *phandle;

    int instrumentid;

    int shape = 0;
    t_calcephcharvalue frame;
    double vector[3];
    double *arraybounds;

    calceph_mex_checknbargs(nlhs, nrhs, 5, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    instrumentid = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
    {
        res = bounds = calceph_getfov(phandle, instrumentid, &shape, frame, vector, NULL, 0);
        if (bounds > 0)
        {
            arraybounds = (double *) mxMalloc(sizeof(double) * bounds * 3);
            if (arraybounds == NULL)
                mexErrMsgTxt("Not enough memory to store values");
            res = calceph_getfov(phandle, instrumentid, &shape, frame, vector, arraybounds, bounds);
            plhs[4] = calceph_convert_array_double_to_mex(arraybounds, 3 * bounds);
            mxFree(arraybounds);
        }
    }
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(shape);
    plhs[2] = calceph_convert_cstring_to_mex(frame);
    plhs[3] = calceph_convert_array_double_to_mex(vector, 3);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_to_cal for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = year
  plhs[2] = month
  plhs[3] = day
  plhs[4] = hour
  plhs[5] = minute
  plhs[6] = second
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = timescale
  prhs[3] = jd0
  prhs[4] = jdfrac
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_to_cal(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int timescale, year = 0, month = 0, day = 0, hour = 0, minute = 0;
    double sec = 0.0, jd0, jdfrac;

    calceph_mex_checknbargs(nlhs, nrhs, 7, 5);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    timescale = calceph_convert_mex_to_integer(prhs[2]);
    jd0 = calceph_convert_mex_to_double(prhs[3]);
    jdfrac = calceph_convert_mex_to_double(prhs[4]);
    if (phandle != NULL)
        res = calceph_time_jd_to_cal(phandle, timescale, jd0, jdfrac, &year, &month, &day, &hour, &minute, &sec);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(year);
    plhs[2] = calceph_convert_integer_to_mex(month);
    plhs[3] = calceph_convert_integer_to_mex(day);
    plhs[4] = calceph_convert_integer_to_mex(hour);
    plhs[5] = calceph_convert_integer_to_mex(minute);
    plhs[6] = calceph_convert_double_to_mex(sec);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_cal_to_jd for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = timescale
  plhs[3] = year
  plhs[4] = month
  plhs[5] = day
  plhs[6] = hour
  plhs[7] = minute
  plhs[8] = second
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_cal_to_jd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int timescale, year, month, day, hour, minute;
    double sec, jd0 = 0, jdfrac = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 9);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    timescale = calceph_convert_mex_to_integer(prhs[2]);
    year = calceph_convert_mex_to_integer(prhs[3]);
    month = calceph_convert_mex_to_integer(prhs[4]);
    day = calceph_convert_mex_to_integer(prhs[5]);
    hour = calceph_convert_mex_to_integer(prhs[6]);
    minute = calceph_convert_mex_to_integer(prhs[7]);
    sec = calceph_convert_mex_to_double(prhs[8]);
    if (phandle != NULL)
        res = calceph_time_cal_to_jd(phandle, timescale, year, month, day, hour, minute, sec, &jd0, &jdfrac);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0);
    plhs[2] = calceph_convert_double_to_mex(jdfrac);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_str_to_jd for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = timescale
  plhs[3] = strdate
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_str_to_jd(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int timescale;
    double jd0 = 0, jdfrac = 0;
    char *strdate;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    timescale = calceph_convert_mex_to_integer(prhs[2]);
    strdate = calceph_convert_mex_to_cstring(prhs[3]);
    if (phandle != NULL)
        res = calceph_time_str_to_jd(phandle, timescale, strdate, &jd0, &jdfrac);

    mxFree(strdate);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0);
    plhs[2] = calceph_convert_double_to_mex(jdfrac);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_set_relationship_tt_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = timescale
  plhs[3] = strdate
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_set_relationship_tt_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int model;

    calceph_mex_checknbargs(nlhs, nrhs, 1, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    model = calceph_convert_mex_to_integer(prhs[2]);
    if (phandle != NULL)
        res = calceph_time_set_relationship_tt_tdb(phandle, model);

    plhs[0] = calceph_convert_integer_to_mex(res);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_tdb_to_jd_tt for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0_tt
  prhs[2] = jdfrac_tt
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = jd0_tdb
  plhs[3] = jdfrac_tdb
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tdb_to_jd_tt(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tt = 0, jdfrac_tt = 0;
    double jd0_tdb, jdfrac_tdb;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0_tdb = calceph_convert_mex_to_double(prhs[2]);
    jdfrac_tdb = calceph_convert_mex_to_double(prhs[3]);

    if (phandle != NULL)
        res = calceph_time_jd_tdb_to_jd_tt(phandle, jd0_tdb, jdfrac_tdb, &jd0_tt, &jdfrac_tt);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tt);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tt);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_tt_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0_tdb
  prhs[2] = jdfrac_tdb
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = jd0_tt
  plhs[3] = jdfrac_tt
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tt_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tt, jdfrac_tt;
    double jd0_tdb = 0, jdfrac_tdb = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0_tt = calceph_convert_mex_to_double(prhs[2]);
    jdfrac_tt = calceph_convert_mex_to_double(prhs[3]);

    if (phandle != NULL)
        res = calceph_time_jd_tt_to_jd_tdb(phandle, jd0_tt, jdfrac_tt, &jd0_tdb, &jdfrac_tdb);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_tdb_to_jd_tcb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0_tcb
  prhs[2] = jdfrac_tcb
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = jd0_tdb
  plhs[3] = jdfrac_tdb
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tdb_to_jd_tcb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tcb = 0, jdfrac_tcb = 0;
    double jd0_tdb, jdfrac_tdb;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0_tdb = calceph_convert_mex_to_double(prhs[2]);
    jdfrac_tdb = calceph_convert_mex_to_double(prhs[3]);

    if (phandle != NULL)
        res = calceph_time_jd_tdb_to_jd_tcb(phandle, jd0_tdb, jdfrac_tdb, &jd0_tcb, &jdfrac_tcb);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tcb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tcb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_tcb_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0_tdb
  prhs[2] = jdfrac_tdb
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = jd0_tcb
  plhs[3] = jdfrac_tcb
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tcb_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tcb, jdfrac_tcb;
    double jd0_tdb = 0, jdfrac_tdb = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0_tcb = calceph_convert_mex_to_double(prhs[2]);
    jdfrac_tcb = calceph_convert_mex_to_double(prhs[3]);

    if (phandle != NULL)
        res = calceph_time_jd_tcb_to_jd_tdb(phandle, jd0_tcb, jdfrac_tcb, &jd0_tdb, &jdfrac_tdb);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_str_any_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  plhs[2] = strdate
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_str_any_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tdb = 0, jdfrac_tdb = 0;
    char *strdate;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    strdate = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_time_str_any_to_jd_tdb(phandle, strdate, &jd0_tdb, &jdfrac_tdb);

    mxFree(strdate);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_str_utc_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  plhs[2] = strdate
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_str_utc_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tdb = 0, jdfrac_tdb = 0;
    char *strdate;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 3);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    strdate = calceph_convert_mex_to_cstring(prhs[2]);
    if (phandle != NULL)
        res = calceph_time_str_utc_to_jd_tdb(phandle, strdate, &jd0_tdb, &jdfrac_tdb);

    mxFree(strdate);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_jd_tdb_to_cal_utc for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = year
  plhs[2] = month
  plhs[3] = day
  plhs[4] = hour
  plhs[5] = minute
  plhs[6] = second
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = jd0_tdb
  prhs[3] = jdfrac_tdb
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tdb_to_cal_utc(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int year = 0, month = 0, day = 0, hour = 0, minute = 0;
    double sec = 0, jd0_tdb, jdfrac_tdb;

    calceph_mex_checknbargs(nlhs, nrhs, 7, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    jd0_tdb = calceph_convert_mex_to_double(prhs[2]);
    jdfrac_tdb = calceph_convert_mex_to_double(prhs[3]);
    if (phandle != NULL)
        res = calceph_time_jd_tdb_to_cal_utc(phandle, jd0_tdb, jdfrac_tdb, &year, &month, &day, &hour, &minute, &sec);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_integer_to_mex(year);
    plhs[2] = calceph_convert_integer_to_mex(month);
    plhs[3] = calceph_convert_integer_to_mex(day);
    plhs[4] = calceph_convert_integer_to_mex(hour);
    plhs[5] = calceph_convert_integer_to_mex(minute);
    plhs[6] = calceph_convert_double_to_mex(sec);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_cal_utc_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0_tdb
  prhs[2] = jdfrac_tdb
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = timescale
  plhs[3] = year
  plhs[4] = month
  plhs[5] = day
  plhs[6] = hour
  plhs[7] = minute
  plhs[8] = second
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_cal_utc_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int year, month, day, hour, minute;
    double sec, jd0_tdb = 0, jdfrac_tdb = 0;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 8);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    year = calceph_convert_mex_to_integer(prhs[2]);
    month = calceph_convert_mex_to_integer(prhs[3]);
    day = calceph_convert_mex_to_integer(prhs[4]);
    hour = calceph_convert_mex_to_integer(prhs[5]);
    minute = calceph_convert_mex_to_integer(prhs[6]);
    sec = calceph_convert_mex_to_double(prhs[7]);
    if (phandle != NULL)
        res = calceph_time_cal_utc_to_jd_tdb(phandle, year, month, day, hour, minute, sec, &jd0_tdb, &jdfrac_tdb);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call calceph_time_str_spacecraft_clock_to_jd_tdb for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  prhs[1] = jd0
  prhs[2] = jdfrac
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = target
  plhs[3] = strdate
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_str_spacecraft_clock_to_jd_tdb(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    double jd0_tdb = 0, jdfrac_tdb = 0;
    char *strdate;
    int target;

    calceph_mex_checknbargs(nlhs, nrhs, 3, 4);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    target = calceph_convert_mex_to_integer(prhs[2]);
    strdate = calceph_convert_mex_to_cstring(prhs[3]);
    if (phandle != NULL)
        res = calceph_time_str_spacecraft_clock_to_jd_tdb(phandle, target, strdate, &jd0_tdb, &jdfrac_tdb);

    mxFree(strdate);
    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_double_to_mex(jd0_tdb);
    plhs[2] = calceph_convert_double_to_mex(jdfrac_tdb);
}

/*--------------------------------------------------------------------------*/
/*! call time_jd_tdb_to_str_spacecraft_clock for a single value.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments
  plhs[0] = return code
  plhs[1] = strdate
  @param nrhs (in) right argument
  @param prhs (in) right arguments :
  prhs[1] = ephemeris descriptor
  prhs[2] = target
  prhs[3] = jd0_tdb
  prhs[4] = jdfrac_tdb
*/
/*--------------------------------------------------------------------------*/
static void calceph_mex_time_jd_tdb_to_str_spacecraft_clock(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int res = 0;

    t_calcephbin *phandle;

    int target;
    double jd0_tdb, jdfrac_tdb;
    t_calcephcharvalue strdate;

    strdate[0] = '\0';

    calceph_mex_checknbargs(nlhs, nrhs, 2, 5);
    phandle = calceph_convert_integer_to_handle(prhs[1]);
    target = calceph_convert_mex_to_integer(prhs[2]);
    jd0_tdb = calceph_convert_mex_to_double(prhs[3]);
    jdfrac_tdb = calceph_convert_mex_to_double(prhs[4]);
    if (phandle != NULL)
        res = calceph_time_jd_tdb_to_str_spacecraft_clock(phandle, target, jd0_tdb, jdfrac_tdb, strdate);

    plhs[0] = calceph_convert_integer_to_mex(res);
    plhs[1] = calceph_convert_cstring_to_mex(strdate);
}

/*--------------------------------------------------------------------------*/
/*! MEX entry point.

  @param nlhs (in) left argument
  @param plhs (inout) left arguments : plhs[0] = ephemris descriptor
  @param nrhs (in) right argument
  @param prhs (in) right arguments : prhs[1] = string
*/
/*--------------------------------------------------------------------------*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    char *caction;

    enum emexaction eaction;

    /* mexPrintf ("I have %d inputs and %d outputs\n", nrhs, nlhs); */

    /* check that the specified action */
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgTxt("First input must be an action string : 'open', 'close', .... ");

    caction = mxArrayToString(prhs[0]);
    eaction = findaction(caction);
    mxFree(caction);
    if (eaction == ACTION_INVALID)
        mexErrMsgTxt("First input must be a valid action string : 'open', 'close', .... ");

    switch (eaction)
    {
        case ACTION_OPEN:
            calceph_mex_open(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_CLOSE:
            calceph_mex_close(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_PREFETCH:
            calceph_mex_prefetch(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_ISTHREADSAFE:
            calceph_mex_isthreadsafe(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_COMPUTE:
            calceph_mex_compute(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_COMPUTE_UNIT:
            calceph_mex_compute_unit(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_COMPUTE_ORDER:
            calceph_mex_compute_order(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_ORIENT_UNIT:
            calceph_mex_orient_unit(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_ORIENT_ORDER:
            calceph_mex_orient_order(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_ROTANGMOM_UNIT:
            calceph_mex_rotangmom_unit(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_ROTANGMOM_ORDER:
            calceph_mex_rotangmom_order(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANT:
            calceph_mex_getconstant(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTSD:
            calceph_mex_getconstantsd(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTVD:
            calceph_mex_getconstantvd(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTSS:
            calceph_mex_getconstantss(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTVS:
            calceph_mex_getconstantvs(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTCOUNT:
            calceph_mex_getconstantcount(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETCONSTANTINDEX:
            calceph_mex_getconstantindex(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETIDBYNAME:
            calceph_mex_getidbyname(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETNAMEBYIDSS:
            calceph_mex_getnamebyidss(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETPOSITIONRECORDCOUNT:
            calceph_mex_getpositionrecordcount(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETPOSITIONRECORDINDEX:
            calceph_mex_getpositionrecordindex(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETPOSITIONRECORDINDEX2:
            calceph_mex_getpositionrecordindex2(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETORIENTRECORDCOUNT:
            calceph_mex_getorientrecordcount(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETORIENTRECORDINDEX:
            calceph_mex_getorientrecordindex(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETORIENTRECORDINDEX2:
            calceph_mex_getorientrecordindex2(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETTIMESCALE:
            calceph_mex_gettimescale(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETTIMESPAN:
            calceph_mex_gettimespan(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETFILEVERSION:
            calceph_mex_getfileversion(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_SETERRORHANDLER:
            calceph_mex_seterrorhandler(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETVERSION_STR:
            calceph_mex_getversion_str(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETMAXSUPPORTEDORDER:
            calceph_mex_getmaxsupportedorder(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TO_CAL:
            calceph_mex_time_jd_to_cal(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_CAL_TO_JD:
            calceph_mex_time_cal_to_jd(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_STR_TO_JD:
            calceph_mex_time_str_to_jd(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_SET_RELATION_SHIP_TT_TDB:
            calceph_mex_time_set_relationship_tt_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TDB_TO_JD_TT:
            calceph_mex_time_jd_tdb_to_jd_tt(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TT_TO_JD_TDB:
            calceph_mex_time_jd_tt_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TDB_TO_JD_TCB:
            calceph_mex_time_jd_tdb_to_jd_tcb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TCB_TO_JD_TDB:
            calceph_mex_time_jd_tcb_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_CAL_UTC_TO_JD_TDB:
            calceph_mex_time_cal_utc_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TDB_TO_CAL_UTC:
            calceph_mex_time_jd_tdb_to_cal_utc(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_STR_UTC_TO_JD_TDB:
            calceph_mex_time_str_utc_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_STR_ANY_TO_JD_TDB:
            calceph_mex_time_str_any_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_SPACECRAFT_CLOCK_TO_JD_TDB:
            calceph_mex_time_str_spacecraft_clock_to_jd_tdb(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_TIME_JD_TDB_TO_SPACECRAFT_CLOCK:
            calceph_mex_time_jd_tdb_to_str_spacecraft_clock(nlhs, plhs, nrhs, prhs);
            break;
        case ACTION_GETFOV:
            calceph_mex_getfov(nlhs, plhs, nrhs, prhs);
            break;
        default:
            break;
    }
}
