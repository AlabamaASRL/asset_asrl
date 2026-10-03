.. _`Instrument functions`:

.. include:: calceph.instrument_intro.rst


.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

calceph_getfov
~~~~~~~~~~~~~~

.. c:function:: int calceph_getfov(t_calcephbin *eph, int instrumentid, int *shape, t_calcephcharvalue frame, double vector[3], double *arraybounds, int nbounds)

    :param  eph: |arg_eph|
    :param instrumentid: Instrument identifier
    :param shape: FOV shape
    :param frame: Reference frame name
    :param vector: Boresight vector
    :param arraybounds: | Array of boundary vectors
                        | The length of the array has to be greater than 3*nbounds. (because each vector has 3 components)
                        | can be NULL to query nbounds necessary 
    :param nbounds: Maximum number of boundary vectors
    :return: 0 if an error occurs, otherwise the number of boundary vectors

.. include:: calceph.instrument_getfov.rst


