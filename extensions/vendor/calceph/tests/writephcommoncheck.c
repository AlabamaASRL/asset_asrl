/*-----------------------------------------------------------------*/
/*!
  \file writephcommoncheck.c
  \brief common functions used by the writeph testsuite.

  \author  A. Durst, M. Gastineau
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

#include "writephcommoncheck.h"
#include "calcephthread.h"
#include <stdlib.h>

/* write one half of a segment in parallel, half_index (0 or 1) says which half */
static void *write_half_par(struct params *p, int half_index)
{
    int first_index = 0;
    int data_shift = 0;

    if (half_index)
    {
        first_index = p->n / 2;
        data_shift = p->size / 2;
    }
    if (p->file_type == 0)
    {
        switch (p->data_type)
        {
            case 2:
                if (!writeph_spk2_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 3:
                if (!writeph_spk3_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 8:
                if (!writeph_spk8_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 9:
                if (!writeph_spk9_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift,
                     p->epochs + first_index))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 12:
                if (!writeph_spk12_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 13:
                if (!writeph_spk13_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift,
                     p->epochs + first_index))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 102:
                if (!writeph_spk102_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (second lines)\n");
                    return NULL;
                }
                break;
            case 103:
                if (!writeph_spk103_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (second lines)\n");
                    return NULL;
                }
                break;
            default:
                printf("Error: write_first: unkown type\n");
        }
    }
    else if (p->file_type == 1)
    {
        switch (p->data_type)
        {
            case 2:
                if (!writeph_pck2_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 3:
                if (!writeph_pck3_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (first lines)\n");
                    return NULL;
                }
                break;
            case 102:
                if (!writeph_pck102_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (second lines)\n");
                    return NULL;
                }
                break;
            case 103:
                if (!writeph_pck103_par_write
                    (p->w_eph, p->reservation, p->target_index, first_index, p->n / 2, p->data + data_shift))
                {
                    printf("Error: writing the coefficients failed (second lines)\n");
                    return NULL;
                }
                break;
            default:
                printf("Error: write_first: unkown type\n");
        }
    }
    else
    {
        printf("Error: Unkown file type\n");
    }
    return (void *) 1;
}

define_fn_thrd_start(write_first, arg)
{
    struct params *p = (struct params *) arg;

    return write_half_par(p, 0) != NULL ? calceph_thrd_success : calceph_thrd_error;
}

define_fn_thrd_start(write_second, arg)
{
    struct params *p = (struct params *) arg;

    return write_half_par(p, 1) != NULL ? calceph_thrd_success : calceph_thrd_error;
}

void write_two_threads(t_writephbin *eph, int reservation, int target_index, double *data, double *epochs, int n,
                       int size, int data_type, int file_type)
{
    /* write coefficients for target 299 with 2 threads */
#define NTHREADS 2
    thrd_start_t fns[NTHREADS];
    void *pargs[NTHREADS];

    struct params p;

    p.w_eph = eph;
    p.reservation = reservation;
    p.target_index = target_index;
    p.data = data;
    p.epochs = epochs;
    p.n = n;
    p.size = size;
    p.data_type = data_type;
    p.file_type = file_type;

    fns[0] = write_first;
    fns[1] = write_second;
    pargs[0] = &p;
    pargs[1] = &p;

    if (calceph_thread_start_and_join(NTHREADS, fns, pargs) == 1)
    {
        printf("calceph_thread_start_and_join fails\n");
        exit(1);
    }

}

static double myabs(double x)
{
    if (x != x)
        return 1E300;
    return (x > 0 ? x : -x);
}

/* check the coordinates interpolated from eph for every date and ref coordinates in the reference file */
int writeph_check_polynomials(t_calcephbin *eph, const char *ref_filename, int target, int center, double threshold,
                              int is_velocity)
{
    int unit = CALCEPH_UNIT_KM + CALCEPH_USE_NAIFID + CALCEPH_UNIT_SEC;

    /* open the reference file */
    FILE *ref = tests_fopen(ref_filename, "r");

    if (!ref)
    {
        printf("writeph_check_polynomials: error when opening the reference file %s\n", ref_filename);
        return 0;
    }

    /* read every line of the reference file */
    double line[7];
    double PV[6];
    int n = is_velocity ? 6 : 3;
    int j;

    while (fscanf
           (ref, "%lf %lf %lf %lf %lf %lf %lf", &line[0], &line[1], &line[2], &line[3], &line[4], &line[5],
            &line[6]) == 7)
    {
        /* interpolate at the current reference date */
        if (!calceph_compute_unit(eph, line[0], 0, target, center, unit, PV))
        {
            printf("writeph_check_polynomials: error when interpolating the coordinates for target %d at date %.17g\n",
                   target, line[0]);
            fclose(ref);
            return 0;
        }

        /* compare the result coordinates with the reference ones */
        for (j = 0; j < n; j++)
        {
            if (myabs((line[j + 1] - PV[j]) / line[j + 1]) > threshold)
            {
                printf("writeph_check_polynomials: error on coordinate %d (time=%.17g): %.17g (expected %.17g)\n", j,
                       line[0], PV[j], line[j + 1]);
                fclose(ref);
                return 0;
            }
        }
    }

    fclose(ref);
    return 1;
}

int writeph_check_states(t_calcephbin *eph, const char *ref_filename, int target, int center, double threshold)
{
    int unit = CALCEPH_UNIT_SEC + CALCEPH_UNIT_KM + CALCEPH_USE_NAIFID;

    /* open the reference file */
    FILE *ref = tests_fopen(ref_filename, "r");

    if (!ref)
    {
        printf("writeph_check_states: error when opening the reference file %s\n", ref_filename);
        return 0;
    }

    /* read every line of the reference file */
    double line[7];
    double PV[6];
    int j;

    while (fscanf
           (ref, "%lf %lf %lf %lf %lf %lf %lf", &line[0], &line[1], &line[2], &line[3], &line[4], &line[5],
            &line[6]) == 7)
    {
        /* interpolate at the current reference date */

        if (!calceph_compute_unit(eph, line[0], 0, target, center, unit, PV))
        {
            printf("writeph_check_states: error when interpolating the coordinates for target %d at date %.17g\n",
                   target, line[0]);
            fclose(ref);
            return 0;
        }

        /* compare the result coordinates with the reference ones */
        for (j = 0; j < 6; j++)
        {
            if (myabs((line[j + 1] - PV[j]) / line[j + 1]) > threshold)
            {
                printf("writeph_check_states: error on coordinate %d (time=%.17g): %.17g (expected %.17g)\n", j,
                       line[0], PV[j], line[j + 1]);
                fclose(ref);
                return 0;
            }
        }
    }

    fclose(ref);
    return 1;
}
