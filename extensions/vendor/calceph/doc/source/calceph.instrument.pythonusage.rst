.. _`Instrument functions`:

.. include:: calceph.instrument_intro.rst


.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

calcephpy.CalcephBin.getfov
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: calcephpy.CalcephBin.getfov(instrumentid) -> shape,  frame, vector, arraybounds

    :param int instrumentid: Instrument identifier
    :return: FOV shape, Reference frame name, Boresight vector, Array of boundary vectors (3 components)
    :rtype: int, str, np.array(3,dtype=float64), np.array(3, nbounds, dtype=float64)

.. include:: calceph.instrument_getfov.rst


