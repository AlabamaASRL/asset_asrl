.. _`Instrument functions`:

.. include:: calceph.instrument_intro.rst


.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

f90calceph_getfov
~~~~~~~~~~~~~~~~~

.. f:function:: function f90calceph_getfov(eph, instrumentid,shape, frame, vector, arraybounds, nbounds)

    :p eph [INTEGER(8), intent(in)]: |arg_eph|
    :p instrumentid [INTEGER, intent(in)]: Instrument identifier
    :p shape [INTEGER, intent(out)]: FOV shape
    :p frame [CHARACTER(len=CALCEPH_MAX_CONSTANTVALUE), intent(out)]: Reference frame name
    :p vector [REAL(8), dimension(1\:3), intent(in)]: Boresight vector
    :p arraybounds [REAL(8), dimension(1\:3*nbounds), intent(out)]: | Array of boundary vectors
                    | The length of the array has to be greater than 3*nbounds. (because each vector has 3 components)
    :p nbounds [INTEGER, intent(in)]: Maximum number of boundary vectors, can be 0 to query nbounds necessary 
    :r f90calceph_getfov: 0 if an error occurs, otherwise the number of boundary vectors
    :rtype f90calceph_getfov: INTEGER
    
.. include:: calceph.instrument_getfov.rst


