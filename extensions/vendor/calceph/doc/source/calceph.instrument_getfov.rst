This function retrieves the instrument Field Of View (FOV) description from an instrument kernel (IK) of the ephemeris file |eph|, and computes the boundary vectors expressed in the instrument frame.

Depending on the FOV specification found in the IK, the function reads a subset of the following keywords (example names use the NAIF convention `INS<id>_...`):

- `INS#_FOV_SHAPE` — shape: `CIRCLE`, `ELLIPSE`, `RECTANGLE`, `POLYGON`.
- `INS#_FOV_FRAME` — name of the reference frame used for the boresight and boundary vectors.
- `INS#_BORESIGHT` — the boresight vector (3 components).
- `INS#_FOV_CLASS_SPEC` — either `ANGLES` or `CORNERS`.

If `FOV_CLASS_SPEC = ANGLES` additional keywords are expected:

- `INS#_FOV_REF_VECTOR` — reference vector defining the reference axis in the boresight plane.
- `INS#_FOV_REF_ANGLE` — half angular extent along the reference axis.
- `INS#_FOV_CROSS_ANGLE` — half angular extent along the axis orthogonal to the reference axis.
- `INS#_FOV_ANGLE_UNITS` — angle units (`DEGREES`, `RADIANS`).

If `FOV_CLASS_SPEC = CORNERS` the IK must provide explicit boundary corner vectors:

- `INS#_FOV_BOUNDARY_CORNERS` — array of 3-component vectors giving the boundary corners.

**Difference between ANGLES and CORNERS:**

- **ANGLES**: The IK describes the FOV by angular half-extents and a reference vector. Boundary vectors are computed from these values. Returned vectors share the same magnitude as the boresight.
- **CORNERS**: The IK provides explicit boundary vectors, which are returned exactly as given. Magnitudes are preserved.

**Coordinate conventions and units:**

- All vectors are expressed in the frame `INS#_FOV_FRAME`.
- Angles may be in `DEGREES` or `RADIANS`; converted internally to radians.
- For `ANGLES`, the boundary vectors preserve boresight magnitude. For `CORNERS`, the magnitudes are taken as-is.

For the **ANGLES** specification, the following formula is used internally to compute FOV boundary vectors. It describes how to rotate a vector :math:`u` inside the plane defined by two vectors :math:`u` and :math:`v`, starting from :math:`u` and turning it by a given angle :math:`\theta`:

.. math::

   \mathbf{u}' = \|\mathbf{u}\| \left( \cos\theta \, \frac{\mathbf{u}}{\|\mathbf{u}\|} + \sin\theta \, \mathbf{Z} \right)

where :math:`\mathbf{Z}` is the normalized projection of :math:`v` onto the plane orthogonal to :math:`u`.

.. math::

   \mathbf{Z} = \frac{\mathbf{v} - \dfrac{(\mathbf{v}\cdot\mathbf{u})}{\|\mathbf{u}\|^2}\,\mathbf{u}}{\left\|\mathbf{v} - \dfrac{(\mathbf{v}\cdot\mathbf{u})}{\|\mathbf{u}\|^2}\,\mathbf{u} \right\|}

This rotation is the core operation used when computing the boundary vectors of a FOV.

For the **CORNERS** specification, vectors are returned as given in the IK.

The possible values for **shape** are :

+------------------------------------+-------------------------+
| value                              |            meaning      |
+====================================+=========================+
| 1                                  | Polygon                 |
+------------------------------------+-------------------------+
| 2                                  | Rectangle               |
+------------------------------------+-------------------------+
| 3                                  | Circle                  |
+------------------------------------+-------------------------+
| 4                                  | Ellipse                 |
+------------------------------------+-------------------------+


.. ifconfig:: calcephapi in ('C')

   If the argument **arraybounds** is **NULL**, the function does not write any boundary vector.
   Instead, it returns the number of boundary vectors required for the given instrument and
   FOV definition.

   This allows the caller to query the required size first, allocate the appropriate array,
   and then call the function again to retrieve the actual boundary vectors.


.. ifconfig:: calcephapi in ('F90', 'F2003')

   If the argument **nbounds** is **0**, the function does not write any boundary vector.
   Instead, it returns the number of boundary vectors required for the given instrument and
   FOV definition.

   This allows the caller to query the required size first, allocate the appropriate array,
   and then call the function again to retrieve the actual boundary vectors.

   Trailing blanks are added the name of the frame.


The following example prints the shape, the frame, and the boundary vector for
a circular FOV

    
    
.. include:: examples/instrument_getfov.rst
