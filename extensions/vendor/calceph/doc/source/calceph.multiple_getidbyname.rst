
The function is not sensitive to the case of the characters of the string *name*. Leading and trailing spaces are ignored. At least one space character is required between the words. For example, 'Saturn Barycenter' is the same as ' Saturn Barycenter ', but is different from 'SaturnBarycenter'.

The library has a default mapping listed in the list :ref:`NAIF identification numbers`. 
The mapping name/id may be overriden by the text constants *NAIF_BODY_CODE* and *NAIF_BODY_NAME* in the 'SPICE' ephemeris files.




The following example prints the identification number of the Moon.

.. include:: examples/multiple_getidbyname.rst