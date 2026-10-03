KPL/FK
 
 
\begintext
          testephem ephemerids file.
 
define the frame MOON_ME_testephem.
define the frame MOON_ME_testephem.
associate the frame MOON_ME_testephem to the Moon
 
MOON_ME_testephem is the name of the lunar principal axes reference frame defined by testephem planetary ephemeris.
 
        \begindata
        OBJECT_MOON_FRAME =  'MOON_ME_INPOP'

        FRAME_MOON_ME_INPOP = 1900302
        FRAME_1900302_NAME = 'MOON_ME_INPOP'
        FRAME_1900302_CLASS = 4
        FRAME_1900302_CLASS_ID = 1900302
        FRAME_1900302_CENTER = 301
        
        TKFRAME_1900302_SPEC            = 'ANGLES'
        TKFRAME_1900302_RELATIVE        = 'MOON_PA_INPOP'
        TKFRAME_1900302_ANGLES          = (  1500     2300     46000   )
        TKFRAME_1900302_AXES            = (   3,        2,        1       )
        TKFRAME_1900302_UNITS           = 'ARCSECONDS'
 
 
        FRAME_MOON_PA_INPOP = 1900301
        FRAME_1900301_NAME = 'MOON_PA_INPOP'
        FRAME_1900301_CLASS = 2
        FRAME_1900301_CLASS_ID = 1900301
        FRAME_1900301_CENTER = 301