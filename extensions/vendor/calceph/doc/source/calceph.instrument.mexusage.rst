.. _`Instrument functions`:

.. include:: calceph.instrument_intro.rst


.. @c %----------------------------------------------------------------------------

.. include:: replace.rst

CalcephBin.getfov
~~~~~~~~~~~~~~~~~

.. mat:method:: CalcephBin.getfov(instrumentid) -> shape,  frame, vector, arraybounds

    :param int instrumentid: Instrument identifier
    :return: FOV shape, Reference frame name, Boresight vector, Array of boundary vectors (3 components)
    :rtype: int, str, vector of 3 floats, vector of floats (3 components per vector)

.. include:: calceph.instrument_getfov.rst


