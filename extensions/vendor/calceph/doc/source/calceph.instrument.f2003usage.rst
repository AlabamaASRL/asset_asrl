.. _`Instrument functions`:

.. include:: calceph.instrument_intro.rst


.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

calceph_getfov
~~~~~~~~~~~~~~

.. f:function:: function calceph_getfov(eph, instrumentid,shape, frame, vector, arraybounds, nbounds)  BIND(C)

    :param eph [TYPE(C_PTR), VALUE, intent(in)]: |arg_eph|
    :p instrumentid [INTEGER(C_INT), VALUE, intent(in)]: Instrument identifier
    :p shape [INTEGER(C_INT), intent(out)]: FOV shape
    :p frame [CHARACTER(len=1,kind=C_CHAR), dimension(CALCEPH_MAX_CONSTANTVALUE), intent(out)]: Reference frame name
    :p vector [REAL(C_DOUBLE), dimension(1\:3), intent(out)]: Boresight vector
    :p arraybounds [REAL(C_DOUBLE), dimension(1\:3*nbounds), intent(out)]: | Array of boundary vectors
                    | The length of the array has to be greater than 3*nbounds. (because each vector has 3 components)
                    | can be NULL to query nbounds necessary 
    :p nbounds [INTEGER(C_INT), VALUE, intent(in)]: Maximum number of boundary vectors
    :r calceph_getfov: 0 if an error occurs, otherwise the number of boundary vectors
    :rtype calceph_getfov: INTEGER(C_INT)
    
.. include:: calceph.instrument_getfov.rst


