/*-----------------------------------------------------------------*/
/*!
  \file calcephgetfov.c
  \brief Retrieve the Field of View (FOV) of an instrument

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
#if HAVE_STRING_H
#include <string.h>
#endif

#include "calceph.h"
#include "util.h"

#define CALCEPH_MAXBOUND 1024

enum calceph_shape
{
    CALCEPH_SHAPE_POLYGON = 1,
    CALCEPH_SHAPE_RECTANGLE,
    CALCEPH_SHAPE_CIRCLE,
    CALCEPH_SHAPE_ELLIPSE
};

/*--------------------------------------------------------------------------*/
/* private functions */
/*--------------------------------------------------------------------------*/
static double norm_vector(double u[3]);

static void cross_vector(double u[3], double v[3], double res[3]);

static void rotate_vector(double u[3], double v[3], double angle, double res[3]);

static void compute_bounds_circle(double vector[3], double ref_vector[3], double ref_angle, double *arraybounds);

static void compute_bounds_rectangle(double vector[3], double ref_vector[3],
                                     double ref_angle, double cross_angle, double *arraybounds);

static void compute_bounds_ellipse(double vector[3], double ref_vector[3], double ref_angle,
                                   double cross_angle, double *arraybounds);


/*--------------------------------------------------------------------------*/
/*! compute the Euclidean norm of a 3D vector

   Formula:
      \|\mathbf{u}\| = \sqrt{u_x^2 + u_y^2 + u_z^2}

   @return the norm of the vector

   @param u (in) 3D vector
*/
/*--------------------------------------------------------------------------*/
static double norm_vector(double u[3])
{
    return sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

/*--------------------------------------------------------------------------*/
/*! compute the cross product of two 3D vectors

   Formula:
      \mathbf{res} = \mathbf{u} \times \mathbf{v} =
      \begin{pmatrix}
         u_y v_z - u_z v_y \\
         u_z v_x - u_x v_z \\
         u_x v_y - u_y v_x
      \end{pmatrix}

   @param u   (in)  first 3D vector
   @param v   (in)  second 3D vector
   @param res (out) resulting 3D vector = u × v
*/
/*--------------------------------------------------------------------------*/
static void cross_vector(double u[3], double v[3], double res[3])
{
    res[0] = u[1] * v[2] - u[2] * v[1];
    res[1] = u[2] * v[0] - u[0] * v[2];
    res[2] = u[0] * v[1] - u[1] * v[0];
}

/*--------------------------------------------------------------------------*/
/*! rotate the vector u in the plane defined by vectors u and v by a given angle

   Formula (rotation in the u-v plane):
      \mathbf{u}_{\text{rot}} =
         \|\mathbf{u}\| \left(
            \cos\theta \frac{\mathbf{u}}{\|\mathbf{u}\|} +
            \sin\theta \frac{\mathbf{v} - (\mathbf{v} \cdot \mathbf{u}) \mathbf{u}/\|\mathbf{u}\|^2}{\|\mathbf{v} - (\mathbf{v} \cdot \mathbf{u}) \mathbf{u}/\|\mathbf{u}\|^2\|}
         \right)

   @param u     (in)  vector to rotate
   @param v     (in)  reference vector defining the rotation plane
   @param angle (in)  rotation angle in radians
   @param res   (out) resulting rotated vector
*/
/*--------------------------------------------------------------------------*/
static void rotate_vector(double u[3], double v[3], double angle, double res[3])
{
    double norm_u = norm_vector(u);

    double cos_theta = cos(angle);
    double sin_theta = sin(angle);

    double a[3];
    double b[3];

    double u_dot_v = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

    a[0] = cos_theta * u[0] / norm_u;
    a[1] = cos_theta * u[1] / norm_u;
    a[2] = cos_theta * u[2] / norm_u;

    b[0] = v[0] - u_dot_v * u[0] / (norm_u * norm_u);
    b[1] = v[1] - u_dot_v * u[1] / (norm_u * norm_u);
    b[2] = v[2] - u_dot_v * u[2] / (norm_u * norm_u);

    double norm_b = norm_vector(b);

    b[0] = sin_theta * b[0] / norm_b;
    b[1] = sin_theta * b[1] / norm_b;
    b[2] = sin_theta * b[2] / norm_b;

    res[0] = norm_u * (a[0] + b[0]);
    res[1] = norm_u * (a[1] + b[1]);
    res[2] = norm_u * (a[2] + b[2]);
}

/*--------------------------------------------------------------------------*/
/*! Compute the boundary vector of a circular FOV.

    @param vector      (in)  boresight vector
    @param ref_vector  (in)  reference vector defining the rotation plane
    @param ref_angle   (in)  rotation angle in radians
    @param arraybounds (out) resulting boundary vector (3 entries)
*/
/*--------------------------------------------------------------------------*/
static void compute_bounds_circle(double vector[3], double ref_vector[3], double ref_angle, double *arraybounds)
{
    double rotate_res[3];

    int j;

    rotate_vector(vector, ref_vector, ref_angle, rotate_res);

    for (j = 0; j < 3; j++)
    {
        arraybounds[j] = rotate_res[j];
    }
}

/*--------------------------------------------------------------------------*/
/*! compute the 4 corner vectors of a rectangular FOV

    @param vector      (in)  boresight vector
    @param ref_vector  (in)  reference vector defining the rotation plane
    @param ref_angle   (in)  rotation angle in radians
    @param cross_angle (in)  rotation angle in the perpendicular plane
    @param arraybounds (out) array of 4 corner vectors (3*4 entries)
*/
/*--------------------------------------------------------------------------*/
static void compute_bounds_rectangle(double vector[3], double ref_vector[3], double ref_angle, double cross_angle,
                                     double *arraybounds)
{
    double rotate_res[3];
    double rotate_cross_res[3];
    double cross_vect[3];

    cross_vector(vector, ref_vector, cross_vect);

    rotate_vector(vector, cross_vect, cross_angle, rotate_cross_res);
    rotate_vector(rotate_cross_res, ref_vector, ref_angle, rotate_res);

    arraybounds[0] = rotate_res[0];
    arraybounds[1] = rotate_res[1];
    arraybounds[2] = rotate_res[2];

    rotate_vector(rotate_cross_res, ref_vector, -ref_angle, rotate_res);

    arraybounds[3] = rotate_res[0];
    arraybounds[4] = rotate_res[1];
    arraybounds[5] = rotate_res[2];

    rotate_vector(vector, cross_vect, -cross_angle, rotate_cross_res);
    rotate_vector(rotate_cross_res, ref_vector, -ref_angle, rotate_res);

    arraybounds[6] = rotate_res[0];
    arraybounds[7] = rotate_res[1];
    arraybounds[8] = rotate_res[2];

    rotate_vector(rotate_cross_res, ref_vector, ref_angle, rotate_res);

    arraybounds[9] = rotate_res[0];
    arraybounds[10] = rotate_res[1];
    arraybounds[11] = rotate_res[2];
}

/*--------------------------------------------------------------------------*/
/*! compute the 2 boundary vectors of an elliptical FOV

    Rotates the boresight vector within planes defined by (boresight, reference)
    and (boresight, cross) vectors by the given angles to obtain the ellipse
    boundary vectors.

    @param vector      (in)  boresight vector
    @param ref_vector  (in)  reference vector defining the rotation plane
    @param ref_angle   (in)  rotation angle in radians
    @param cross_angle (in)  rotation angle in the perpendicular plane
    @param arraybounds (out) array of 2 corner vectors (3*2 entries)
*/
/*--------------------------------------------------------------------------*/
static void compute_bounds_ellipse(double vector[3], double ref_vector[3], double ref_angle, double cross_angle,
                                   double *arraybounds)
{
    double rotate_res[3];
    double cross_vect[3];

    cross_vector(vector, ref_vector, cross_vect);

    rotate_vector(vector, ref_vector, ref_angle, rotate_res);

    arraybounds[0] = rotate_res[0];
    arraybounds[1] = rotate_res[1];
    arraybounds[2] = rotate_res[2];

    rotate_vector(vector, cross_vect, cross_angle, rotate_res);

    arraybounds[3] = rotate_res[0];
    arraybounds[4] = rotate_res[1];
    arraybounds[5] = rotate_res[2];
}

/*--------------------------------------------------------------------------*/
/*! retrieve the Field of View (FOV) vectors for an instrument

    Reads the FOV shape, frame, boresight, angles and boundary corners from
    the ephemeris and computes the boundary vectors.

    The function ensures that output parameters are only modified at the end
    and supports querying the required number of boundary vectors if
    arraybounds=NULL.

    @return number of boundary vectors
    @return 0 on error

    @param eph          (in)  ephemeris object
    @param instrumentid (in) instrument identifier
    @param shape        (out) FOV shape
    @param frame        (out) reference frame name
    @param vector       (out) boresight vector
    @param arraybounds  (out) array of boundary vectors (can be NULL to query count)
    @param nbounds      (in)  maximum number of boundary vectors
*/
/*--------------------------------------------------------------------------*/
int calceph_getfov(t_calcephbin *eph, int instrumentid, int *shape,
                   t_calcephcharvalue frame, double vector[3], double *arraybounds, int nbounds)
{
    if (eph == NULL)
    {
        fatalerror("eph is not initialized\n");
        return 0;
    }

    if (shape == NULL)
    {
        fatalerror("shape pointer cannot be null\n");
        return 0;
    }

    char cname[CALCEPH_MAX_CONSTANTNAME];

    t_calcephcharvalue svalue;

    int with_angles = 0, in_degrees = 0;

    double ref_angle = 0, cross_angle = 0;

    int res = 0;

    double ref_vector[3];

    int tmp_shape;

    t_calcephcharvalue tmp_frame;

    double tmp_vector[3];

    double tmp_arraybounds[CALCEPH_MAXBOUND * 3];

    /*---------------------------------------------------------*/
    /* get the shape */
    /*---------------------------------------------------------*/
    calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_SHAPE", instrumentid);
    if (calceph_getconstantss(eph, cname, svalue) == 0)
    {
        /* ERROR: not found */
        fatalerror("The shape of the instrument %i can not be found\n", instrumentid);
        return 0;
    }

    if (strcmp(svalue, "POLYGON") == 0)
    {
        tmp_shape = CALCEPH_SHAPE_POLYGON;
    }
    else if (strcmp(svalue, "RECTANGLE") == 0)
    {
        tmp_shape = CALCEPH_SHAPE_RECTANGLE;
    }
    else if (strcmp(svalue, "CIRCLE") == 0)
    {
        tmp_shape = CALCEPH_SHAPE_CIRCLE;
    }
    else if (strcmp(svalue, "ELLIPSE") == 0)
    {
        tmp_shape = CALCEPH_SHAPE_ELLIPSE;
    }
    else
    {
        /* ERROR: unsupported value */
        fatalerror("The shape of %i is not supported.\n"
                   "It should be 'CIRCLE', 'ELLIPSE', 'RECTANGLE' or 'POLYGON'", instrumentid);
        return 0;
    }

    /*---------------------------------------------------------*/
    /* get the frame */
    /*---------------------------------------------------------*/
    calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_FRAME", instrumentid);
    if (calceph_getconstantss(eph, cname, tmp_frame) == 0)
    {
        /* ERROR: not found */
        fatalerror("The frame of the instrument %i can not be found\n", instrumentid);
        return 0;
    }

    /*---------------------------------------------------------*/
    /* get the class specification (angles or corners) */
    /*---------------------------------------------------------*/
    calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_CLASS_SPEC", instrumentid);
    if (calceph_getconstantss(eph, cname, svalue) != 0)
    {
        if (strcmp(svalue, "ANGLES") == 0)
            with_angles = 1;
        else if (strcmp(svalue, "CORNERS") != 0)
        {
            /* ERROR: unsupported value */
            fatalerror("Unsupported FOV class spec for %i: %s (should be 'ANGLES' or 'CORNERS')\n",
                       instrumentid, svalue);
            return 0;
        }
    }
    else
    {
        /* When specification is missing, the class is supposed to be 'CORNERS' */
        with_angles = 0;
    }

    /*---------------------------------------------------------*/
    /* get the boresight vector */
    /*---------------------------------------------------------*/
    calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_BORESIGHT", instrumentid);
    if (calceph_getconstantvd(eph, cname, tmp_vector, 3) == 0)
    {
        /* ERROR: not found */
        fatalerror("Boresight vector for %i is missing\n", instrumentid);
        return 0;
    }

    if (tmp_vector[0] == 0.0 && tmp_vector[1] == 0.0 && tmp_vector[2] == 0.0)
    {
        /* ERROR: zero-vector */
        fatalerror("Boresight vector for %i is the zero vector\n", instrumentid);
        return 0;
    }

    if (with_angles == 1)
    {
        /*---------------------------------------------------------*/
        /* get the ref-angle */
        /*---------------------------------------------------------*/
        calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_REF_ANGLE", instrumentid);
        if (calceph_getconstant(eph, cname, &ref_angle) == 0)
        {
            /* ERROR: not found */
            fatalerror("Reference angle missing for %i\n", instrumentid);
            return 0;
        }

        /*---------------------------------------------------------*/
        /* get the cross-angle (if needed) */
        /*---------------------------------------------------------*/
        calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_CROSS_ANGLE", instrumentid);
        if (calceph_getconstant(eph, cname, &cross_angle) == 0 &&
            (tmp_shape != CALCEPH_SHAPE_POLYGON && tmp_shape != CALCEPH_SHAPE_CIRCLE))
        {
            /* ERROR: not found */
            fatalerror("Cross angle missing for %i\n", instrumentid);
            return 0;
        }

        /*---------------------------------------------------------*/
        /* get the units */
        /*---------------------------------------------------------*/
        calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_ANGLE_UNITS", instrumentid);
        if (calceph_getconstantss(eph, cname, svalue) == 0)
        {
            /* ERROR: not found */
            fatalerror("Angle units missing for %i\n", instrumentid);
            return 0;
        }

        if (strcmp(svalue, "DEGREES") == 0)
        {
            in_degrees = 1;
        }
        else if (strcmp(svalue, "RADIANS") != 0)
        {
            /* ERROR: unsupported value */
            fatalerror("Unsupported angle units %s for %i\n", svalue, instrumentid);
            return 0;
        }

        if (in_degrees == 1)
        {
            /* degree to radian */
            ref_angle = ref_angle * M_PI / 180.;
            cross_angle = cross_angle * M_PI / 180.;
        }

        /*---------------------------------------------------------*/
        /* get the ref-vector */
        /*---------------------------------------------------------*/
        calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_REF_VECTOR", instrumentid);
        if (calceph_getconstantvd(eph, cname, ref_vector, 3) == 0)
        {
            /* ERROR: not found */
            fatalerror("Reference vector missing for %i\n", instrumentid);
            return 0;
        }

        if ((ref_vector[0] == 0 && ref_vector[1] == 0 && ref_vector[2] == 0) ||
            (fabs(ref_vector[0] * tmp_vector[0] + ref_vector[1] * tmp_vector[1] + ref_vector[2] * tmp_vector[2]) >
             0.999999))
        {
            /* ERROR: bad ref-vector specifications */
            fatalerror("Bad reference vector spec for %i\n", instrumentid);
            return 0;
        }

        /*---------------------------------------------------------*/
        /* compute bounds vectors */
        /*---------------------------------------------------------*/
        if (tmp_shape == CALCEPH_SHAPE_RECTANGLE)
        {
            if (arraybounds == NULL)
            {
                return 4;
            }

            if (nbounds < 4)
            {
                /* ERROR: arraybounds size not big enough */
                fatalerror("Not enough place in arraybounds for instument %i\n", instrumentid);
                return 0;
            }
            compute_bounds_rectangle(tmp_vector, ref_vector, ref_angle, cross_angle, tmp_arraybounds);
            res = 4;
        }
        else if (tmp_shape == CALCEPH_SHAPE_CIRCLE)
        {
            if (arraybounds == NULL)
            {
                return 1;
            }

            if (nbounds < 1)
            {
                /* ERROR: arraybounds size not big enough */

                fatalerror("Not enough place in arraybounds for instument %i\n", instrumentid);
                return 0;
            }
            compute_bounds_circle(tmp_vector, ref_vector, ref_angle, tmp_arraybounds);
            res = 1;
        }
        else if (tmp_shape == CALCEPH_SHAPE_ELLIPSE)
        {
            if (arraybounds == NULL)
            {
                return 2;
            }

            if (nbounds < 2)
            {
                /* ERROR: arraybounds size not big enough */
                fatalerror("Not enough place in arraybounds for instument %i\n", instrumentid);
                return 0;
            }
            compute_bounds_ellipse(tmp_vector, ref_vector, ref_angle, cross_angle, tmp_arraybounds);
            res = 2;
        }
        else if (*shape == CALCEPH_SHAPE_POLYGON)
        {
            /* ERROR: unsupported with polygon */
            fatalerror("Polygon not supported with ANGLES for %i\n", instrumentid);
            return 0;
        }

    }
    else
    {
        /*---------------------------------------------------------*/
        /* get boundary corners */
        /*---------------------------------------------------------*/

        calceph_snprintf(cname, CALCEPH_MAX_CONSTANTNAME, "INS%d_FOV_BOUNDARY_CORNERS", instrumentid);
        int n_boundary_corners = calceph_getconstantvd(eph, cname, NULL, 0);

        if (n_boundary_corners == 0)
        {
            /* ERROR: not found */
            fatalerror("Boundary corners missing for %i\n", instrumentid);
            return 0;
        }
        if (n_boundary_corners % 3 != 0)
        {
            /* ERROR: bad boundary corners specifications */
            fatalerror("Bad boundary spec for %i: not multiple of 3\n", instrumentid);
            return 0;
        }

        if (arraybounds == NULL)
        {
            return n_boundary_corners / 3;
        }

        if (n_boundary_corners / 3 > nbounds)
        {
            /* ERROR: too many vectors compared to nbounds */
            fatalerror("number of boundary vectors bigger than nbounds for %i\n", instrumentid);
            return 0;
        }

        calceph_getconstantvd(eph, cname, tmp_arraybounds, n_boundary_corners);
        res = n_boundary_corners / 3;
    }

    /*---------------------------------------------------------*/
    /* update outputs */
    /*---------------------------------------------------------*/

    *shape = tmp_shape;
    strcpy(frame, tmp_frame);
    memcpy(vector, tmp_vector, 3 * sizeof(double));
    if (arraybounds != NULL && res > 0)
    {
        memcpy(arraybounds, tmp_arraybounds, res * 3 * sizeof(double));
    }

    return res;
}
