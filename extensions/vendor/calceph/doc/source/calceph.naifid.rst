.. include:: replace.rst

.. _`NAIF identification numbers`:

NAIF identification numbers
===========================

The following predefined values must be used as the target body and origin of the coordinate system with the functions |calceph_compute_unit|, |calceph_orient_unit| |calceph_compute_order| or |calceph_orient_order| if and only if the value |CALCEPH_USE_NAIFID| has been set in the parameter *unit*.

.. ifconfig:: calcephapi in ('C')

    This list is already predefined in the interface file :file:`calceph.h`. 

.. ifconfig:: calcephapi in ('F2003')

    This list is already predefined in the module file :file:`calceph.mod`.

.. ifconfig:: calcephapi in ('F90')
    
    This list is already predefined in the interface file :file:`f90calceph.h`.

.. ifconfig:: calcephapi in ('Python')

    This list is already predefined in the class NaifId (:ref:`NaifId`) of the module calcephpy (:ref:`calcephpy`) for the Python 2/3 interface. Relative to C or Fortran interface, the prefix *NAIFID_* is deleted for the following numbers.  

.. ifconfig:: calcephapi in ('Mex')

    This list is already predefined in the class :ref:`NaifId` of the module :ref:`calcephpy` for the Octave/Matlab interface. Relative to C or Fortran interface, the prefix *NAIFID_* is deleted for the following numbers.  

Sun and planetary barycenters
-----------------------------

+----------------------------------+-----------+--------------------------+
| Predefined Macros                | NAIF ID   | Name                     |
+==================================+===========+==========================+
| NAIFID_SOLAR_SYSTEM_BARYCENTER   | 0         | Solar System Barycenter  |
+----------------------------------+-----------+--------------------------+
| NAIFID_MERCURY_BARYCENTER        | 1         | Mercury Barycenter       |
+----------------------------------+-----------+--------------------------+
| NAIFID_VENUS_BARYCENTER          | 2         | Venus Barycenter         |
+----------------------------------+-----------+--------------------------+
| NAIFID_EARTH_MOON_BARYCENTER     | 3         | Earth-Moon Barycenter    |
+----------------------------------+-----------+--------------------------+
| NAIFID_MARS_BARYCENTER           | 4         | Mars Barycenter          |
+----------------------------------+-----------+--------------------------+
| NAIFID_JUPITER_BARYCENTER        | 5         | Jupiter Barycenter       |
+----------------------------------+-----------+--------------------------+
| NAIFID_SATURN_BARYCENTER         | 6         | Saturn Barycenter        |
+----------------------------------+-----------+--------------------------+
| NAIFID_URANUS_BARYCENTER         | 7         | Uranus Barycenter        |
+----------------------------------+-----------+--------------------------+
| NAIFID_NEPTUNE_BARYCENTER        | 8         | Neptune Barycenter       |
+----------------------------------+-----------+--------------------------+
| NAIFID_PLUTO_BARYCENTER          | 9         | Pluto Barycenter         |
+----------------------------------+-----------+--------------------------+
| NAIFID_SUN                       | 10        | Sun                      |
+----------------------------------+-----------+--------------------------+

Coordinate Time ephemerides 
---------------------------

+---------------------+------------+--------------------------------------------------+
| Predefined Macros   | NAIF ID    | Name                                             |
+=====================+============+==================================================+
| NAIFID_TIME_CENTER  | 1000000000 | center ID for Coordinate Time ephemerides [#f1]_ |
+---------------------+------------+--------------------------------------------------+
| NAIFID_TIME_TTMTDB  | 1000000001 | Coordinate Time ephemeride TT-TDB [#f2]_         |
+---------------------+------------+--------------------------------------------------+
| NAIFID_TIME_TCGMTCB | 1000000002 | Coordinate Time ephemeride TCG-TCB [#f2]_        |
+---------------------+------------+--------------------------------------------------+

.. [#f1] These values must only be used as a center body.
.. [#f2] These values must only be used as a target body.

Planet centers and satellites
-----------------------------

========================= ======== ====================
  Predefined Macros       NAIF ID   Name                
========================= ======== ====================
      NAIFID_MERCURY           199      Mercury            
------------------------- -------- --------------------
------------------------- -------- --------------------
      NAIFID_VENUS             299      Venus               
------------------------- -------- --------------------
------------------------- -------- --------------------
      NAIFID_EARTH             399     Earth               
      NAIFID_MOON              301     Moon                
------------------------- -------- --------------------
------------------------- -------- --------------------
      NAIFID_MARS              499     Mars                
      NAIFID_PHOBOS            401      Phobos             
      NAIFID_DEIMOS            402     Deimos     
------------------------- -------- --------------------
------------------------- -------- --------------------
      NAIFID_JUPITER           599     Jupiter             
      NAIFID_IO                501      Io                 
      NAIFID_EUROPA            502      Europa             
      NAIFID_GANYMEDE          503      Ganymede           
      NAIFID_CALLISTO          504      Callisto           
      NAIFID_AMALTHEA          505      Amalthea           
      NAIFID_HIMALIA           506      Himalia            
      NAIFID_ELARA             507      Elara              
      NAIFID_PASIPHAE          508      Pasiphae           
      NAIFID_SINOPE            509      Sinope             
      NAIFID_LYSITHEA          510      Lysithea           
      NAIFID_CARME             511      Carme              
      NAIFID_ANANKE            512      Ananke             
      NAIFID_LEDA              513      Leda               
      NAIFID_THEBE             514      Thebe              
      NAIFID_ADRASTEA          515      Adrastea           
      NAIFID_METIS             516      Metis              
      NAIFID_CALLIRRHOE        517      Callirrhoe         
      NAIFID_THEMISTO          518      Themisto           
      NAIFID_MEGACLITE         519      Megaclite          
      NAIFID_TAYGETE           520      Taygete            
      NAIFID_CHALDENE          521      Chaldene           
      NAIFID_HARPALYKE         522      Harpalyke          
      NAIFID_KALYKE            523      Kalyke             
      NAIFID_IOCASTE           524      Iocaste            
      NAIFID_ERINOME           525      Erinome            
      NAIFID_ISONOE            526      Isonoe             
      NAIFID_PRAXIDIKE         527      Praxidike          
      NAIFID_AUTONOE           528      Autonoe            
      NAIFID_THYONE            529      Thyone             
      NAIFID_HERMIPPE          530      Hermippe           
      NAIFID_AITNE             531      Aitne              
      NAIFID_EURYDOME          532      Eurydome           
      NAIFID_EUANTHE           533      Euanthe            
      NAIFID_EUPORIE           534      Euporie            
      NAIFID_ORTHOSIE          535      Orthosie           
      NAIFID_SPONDE            536      Sponde             
      NAIFID_KALE              537      Kale               
      NAIFID_PASITHEE          538      Pasithee           
      NAIFID_HEGEMONE          539     Hegemone            
      NAIFID_MNEME             540     Mneme               
      NAIFID_AOEDE             541     Aoede               
      NAIFID_THELXINOE         542     Thelxinoe           
      NAIFID_ARCHE             543     Arche               
      NAIFID_KALLICHORE        544     Kallichore          
      NAIFID_HELIKE            545     Helike              
      NAIFID_CARPO             546     Carpo               
      NAIFID_EUKELADE          547     Eukelade            
      NAIFID_CYLLENE           548     Cyllene             
      NAIFID_KORE              549     Kore                
      NAIFID_HERSE             550     Herse               
      NAIFID_DIA               553     Dia               
------------------------- -------- --------------------
------------------------- -------- --------------------
      NAIFID_SATURN            699     Saturn              
    NAIFID_MIMAS             601      Mimas              
    NAIFID_ENCELADUS         602      Enceladus          
    NAIFID_TETHYS            603      Tethys             
    NAIFID_DIONE             604      Dione              
    NAIFID_RHEA              605      Rhea               
    NAIFID_TITAN             606      Titan              
    NAIFID_HYPERION           607        Hyperion               
    NAIFID_IAPETUS            608        Iapetus               
    NAIFID_PHOEBE             609        Phoebe               
    NAIFID_JANUS              610        Janus               
    NAIFID_EPIMETHEUS         611        Epimetheus               
    NAIFID_HELENE             612        Helene               
    NAIFID_TELESTO            613        Telesto               
    NAIFID_CALYPSO            614        Calypso               
    NAIFID_ATLAS              615        Atlas               
    NAIFID_PROMETHEUS         616        Prometheus               
    NAIFID_PANDORA            617       Pandora               
    NAIFID_PAN                618       Pan               
    NAIFID_YMIR               619       Ymir               
    NAIFID_PAALIAQ            620       Paaliaq               
    NAIFID_TARVOS             621       Tarvos               
    NAIFID_IJIRAQ             622       Ijiraq               
    NAIFID_SUTTUNGR           623       Suttungr               
    NAIFID_KIVIUQ             624       Kiviuq               
    NAIFID_MUNDILFARI         625       Mundilfari               
    NAIFID_ALBIORIX           626       Albiorix               
    NAIFID_SKATHI             627       Skathi              
    NAIFID_ERRIAPUS           628       Erriapus              
    NAIFID_SIARNAQ            629       Siarnaq              
    NAIFID_THRYMR             630       Thrymr              
    NAIFID_NARVI              631       Narvi              
    NAIFID_METHONE            632       Methone              
    NAIFID_PALLENE            633       Pallene              
    NAIFID_POLYDEUCES         634       Polydeuces              
    NAIFID_DAPHNIS            635       Daphnis             
    NAIFID_AEGIR              636       Aegir             
    NAIFID_BEBHIONN           637       Bebhionn             
    NAIFID_BERGELMIR          638       Bergelmir             
    NAIFID_BESTLA             639       Bestla             
    NAIFID_FARBAUTI           640       Farbauti             
    NAIFID_FENRIR             641       Fenrir             
    NAIFID_FORNJOT            642       Fornjot             
    NAIFID_HATI               643       Hati             
    NAIFID_HYROKKIN           644       Hyrokkin             
    NAIFID_KARI               645       Kari             
    NAIFID_LOGE               646       Loge             
    NAIFID_SKOLL              647       Skoll             
    NAIFID_SURTUR             648       Surtur             
    NAIFID_ANTHE              649       Anthe             
    NAIFID_JARNSAXA           650       Jarnsaxa             
    NAIFID_GREIP              651       Greip             
    NAIFID_TARQEQ             652       Tarqeq             
    NAIFID_AEGAEON            653       Aegaeon             
------------------------- -------- --------------------
------------------------- -------- --------------------
    NAIFID_URANUS             799       Uranus             
    NAIFID_ARIEL              701       Ariel               
    NAIFID_UMBRIEL            702       Umbriel               
    NAIFID_TITANIA            703       Titania               
    NAIFID_OBERON             704       Oberon               
    NAIFID_MIRANDA            705       Miranda               
    NAIFID_CORDELIA           706       Cordelia               
    NAIFID_OPHELIA            707       Ophelia               
    NAIFID_BIANCA             708       Bianca               
    NAIFID_CRESSIDA           709       Cressida               
    NAIFID_DESDEMONA          710       Desdemona               
    NAIFID_JULIET             711       Juliet               
    NAIFID_PORTIA             712       Portia               
    NAIFID_ROSALIND           713       Rosalind               
    NAIFID_BELINDA            714       Belinda               
    NAIFID_PUCK               715       Puck               
    NAIFID_CALIBAN            716       Caliban               
    NAIFID_SYCORAX            717       Sycorax               
    NAIFID_PROSPERO           718       Prospero               
    NAIFID_SETEBOS            719       Setebos               
    NAIFID_STEPHANO           720       Stephano               
    NAIFID_TRINCULO           721     Trinculo               
    NAIFID_FRANCISCO          722     Francisco             
    NAIFID_MARGARET           723     Margaret             
    NAIFID_FERDINAND          724     Ferdinand             
    NAIFID_PERDITA            725     Perdita             
    NAIFID_MAB                726     Mab             
    NAIFID_CUPID              727     Cupid             
------------------------- -------- --------------------
------------------------- -------- --------------------
    NAIFID_NEPTUNE            899     Neptune            
    NAIFID_TRITON             801     Triton             
    NAIFID_NEREID             802     Nereid             
    NAIFID_NAIAD              803     Naiad              
    NAIFID_THALASSA           804     Thalassa           
    NAIFID_DESPINA            805     Despina            
    NAIFID_GALATEA            806     Galatea            
    NAIFID_LARISSA            807     Larissa            
    NAIFID_PROTEUS            808     Proteus            
    NAIFID_HALIMEDE           809     Halimede           
    NAIFID_PSAMATHE           810     Psamathe           
    NAIFID_SAO                811     Sao                
    NAIFID_LAOMEDEIA          812     Laomedeia          
    NAIFID_NESO               813      Neso              
------------------------- -------- --------------------
------------------------- -------- --------------------
    NAIFID_PLUTO              999      Pluto             
    NAIFID_CHARON             901      Charon            
    NAIFID_NIX                902      Nix               
    NAIFID_HYDRA              903      Hydra             
    NAIFID_KERBEROS           904      Kerberos             
    NAIFID_STYX               905      Styx             
========================= ======== ====================

Comets
------

===================================== ======== ====================
  Predefined Macros                   NAIF ID   Name                
===================================== ======== ====================
 NAIFID_AREND                          1000001     Arend
 NAIFID_AREND_RIGAUX                   1000002     Arend-Rigaux
 NAIFID_ASHBROOK_JACKSON               1000003     Ashbrook-Jackson
 NAIFID_BOETHIN                        1000004     Boethin
 NAIFID_BORRELLY                       1000005     Borrelly
 NAIFID_BOWELL_SKIFF                   1000006     Bowell-Skiff
 NAIFID_BRADFIELD                      1000007     Bradfield
 NAIFID_BROOKS_2                       1000008     Brooks 2
 NAIFID_BRORSEN_METCALF                1000009     Brorsen-Metcalf
 NAIFID_BUS                            1000010     Bus
 NAIFID_CHERNYKH                       1000011     Chernykh
 NAIFID_CHURYUMOV_GERASIMENKO          1000012     Churyumov-Gerasimenko
 NAIFID_CIFFREO                        1000013     Ciffreo
 NAIFID_CLARK                          1000014     Clark
 NAIFID_COMAS_SOLA                     1000015     Comas Sola
 NAIFID_CROMMELIN                      1000016     Crommelin
 NAIFID_D__ARREST                      1000017     D''Drrest
 NAIFID_DANIEL                         1000018     Daniel
 NAIFID_DE_VICO_SWIFT                  1000019     De Vico-Swift
 NAIFID_DENNING_FUJIKAWA               1000020     Denning-Fujikawa
 NAIFID_DU_TOIT_1                      1000021     Du Toit 1
 NAIFID_DU_TOIT_HARTLEY                1000022     Du Toit-Hartley
 NAIFID_DUTOIT_NEUJMIN_DELPORTE        1000023     Dutoit-Neujmin-Delporte
 NAIFID_DUBIAGO                        1000024     Dubiago
 NAIFID_ENCKE                          1000025     Encke
 NAIFID_FAYE                           1000026     Faye
 NAIFID_FINLAY                         1000027     Finlay
 NAIFID_FORBES                         1000028     Forbes
 NAIFID_GEHRELS_1                      1000029     Gehrels 1
 NAIFID_GEHRELS_2                      1000030     Gehrels 2
 NAIFID_GEHRELS_3                      1000031     Gehrels 3
 NAIFID_GIACOBINI_ZINNER               1000032     Giacobini-Zinner
 NAIFID_GICLAS                         1000033     Giclas
 NAIFID_GRIGG_SKJELLERUP               1000034     Grigg-Skjellerup
 NAIFID_GUNN                           1000035     Gunn
 NAIFID_HALLEY                         1000036     Halley
 NAIFID_HANEDA_CAMPOS                  1000037     Haneda-Campos
 NAIFID_HARRINGTON                     1000038     Harrington
 NAIFID_HARRINGTON_ABELL               1000039     Harrington-Abell
 NAIFID_HARTLEY_1                      1000040     Hartley 1
 NAIFID_HARTLEY_2                      1000041     Hartley 2
 NAIFID_HARTLEY_IRAS                   1000042     Hartley-Iras
 NAIFID_HERSCHEL_RIGOLLET              1000043     Herschel-Rigollet
 NAIFID_HOLMES                         1000044     Holmes
 NAIFID_HONDA_MRKOS_PAJDUSAKOVA        1000045     Honda-Mrkos-Pajdusakova
 NAIFID_HOWELL                         1000046     Howell
 NAIFID_IRAS                           1000047     Iras
 NAIFID_JACKSON_NEUJMIN                1000048     Jackson-Neujmin
 NAIFID_JOHNSON                        1000049     Johnson
 NAIFID_KEARNS_KWEE                    1000050     Kearns-Kwee
 NAIFID_KLEMOLA                        1000051     Klemola
 NAIFID_KOHOUTEK                       1000052     Kohoutek
 NAIFID_KOJIMA                         1000053     Kojima
 NAIFID_KOPFF                          1000054     Kopff
 NAIFID_KOWAL_1                        1000055     Kowal 1
 NAIFID_KOWAL_2                        1000056     Kowal 2
 NAIFID_KOWAL_MRKOS                    1000057     Kowal-Mrkos
 NAIFID_KOWAL_VAVROVA                  1000058     Kowal-Vavrova
 NAIFID_LONGMORE                       1000059     Longmore
 NAIFID_LOVAS_1                        1000060     Lovas 1
 NAIFID_MACHHOLZ                       1000061     Machholz
 NAIFID_MAURY                          1000062     Maury
 NAIFID_NEUJMIN_1                      1000063     Neujmin 1
 NAIFID_NEUJMIN_2                      1000064     Neujmin 2
 NAIFID_NEUJMIN_3                      1000065     Neujmin 3
 NAIFID_OLBERS                         1000066     Olbers
 NAIFID_PETERS_HARTLEY                 1000067     Peters-Hartley
 NAIFID_PONS_BROOKS                    1000068     Pons-Brooks
 NAIFID_PONS_WINNECKE                  1000069     Pons-Winnecke
 NAIFID_REINMUTH_1                     1000070     Reinmuth 1
 NAIFID_REINMUTH_2                     1000071     Reinmuth 2
 NAIFID_RUSSELL_1                      1000072     Russell 1
 NAIFID_RUSSELL_2                      1000073     Russell 2
 NAIFID_RUSSELL_3                      1000074     Russell 3
 NAIFID_RUSSELL_4                      1000075     Russell 4
 NAIFID_SANGUIN                        1000076     Sanguin
 NAIFID_SCHAUMASSE                     1000077     Schaumasse
 NAIFID_SCHUSTER                       1000078     Schuster
 NAIFID_SCHWASSMANN_WACHMANN_1         1000079     Schwassmann-Wachmann 1
 NAIFID_SCHWASSMANN_WACHMANN_2         1000080     Schwassmann-Wachmann 2
 NAIFID_SCHWASSMANN_WACHMANN_3         1000081     Schwassmann-Wachmann 3
 NAIFID_SHAJN_SCHALDACH                1000082     Shajn-Schaldach
 NAIFID_SHOEMAKER_1                    1000083     Shoemaker 1
 NAIFID_SHOEMAKER_2                    1000084     Shoemaker 2
 NAIFID_SHOEMAKER_3                    1000085     Shoemaker 3
 NAIFID_SINGER_BREWSTER                1000086     Singer-Brewster
 NAIFID_SLAUGHTER_BURNHAM              1000087     Slaughter-Burnham
 NAIFID_SMIRNOVA_CHERNYKH              1000088     Smirnova-Chernykh
 NAIFID_STEPHAN_OTERMA                 1000089     Stephan-Oterma
 NAIFID_SWIFT_GEHRELS                  1000090     Swift-Gehrels
 NAIFID_TAKAMIZAWA                     1000091     Takamizawa
 NAIFID_TAYLOR                         1000092     Taylor
 NAIFID_TEMPEL_1                       1000093     Tempel 1
 NAIFID_TEMPEL_2                       1000094     Tempel 2
 NAIFID_TEMPEL_TUTTLE                  1000095     Tempel-Tuttle
 NAIFID_TRITTON                        1000096     Tritton
 NAIFID_TSUCHINSHAN_1                  1000097     Tsuchinshan 1
 NAIFID_TSUCHINSHAN_2                  1000098     Tsuchinshan 2
 NAIFID_TUTTLE                         1000099     Tuttle
 NAIFID_TUTTLE_GIACOBINI_KRESAK        1000100     Tuttle-Giacobini-Kresak
 NAIFID_VAISALA_1                      1000101     Vaisala 1
 NAIFID_VAN_BIESBROECK                 1000102     Van Biesbroeck
 NAIFID_VAN_HOUTEN                     1000103     Van Houten
 NAIFID_WEST_KOHOUTEK_IKEMURA          1000104     West-Kohoutek-Ikemura
 NAIFID_WHIPPLE                        1000105     Whipple
 NAIFID_WILD_1                         1000106     Wild 1
 NAIFID_WILD_2                         1000107     Wild 2
 NAIFID_WILD_3                         1000108     Wild 3
 NAIFID_WIRTANEN                       1000109     Wirtanen
 NAIFID_WOLF                           1000110     Wolf
 NAIFID_WOLF_HARRINGTON                1000111     Wolf-Harrington
 NAIFID_LOVAS_2                        1000112     Lovas 2
 NAIFID_URATA_NIIJIMA                  1000113     Urata-Niijima
 NAIFID_WISEMAN_SKIFF                  1000114     Wiseman-Skiff
 NAIFID_HELIN                          1000115     Helin
 NAIFID_MUELLER                        1000116     Mueller
 NAIFID_SHOEMAKER_HOLT_1               1000117     Shoemaker-Holt 1
 NAIFID_HELIN_ROMAN_CROCKETT           1000118     Helin-Roman-Crockett
 NAIFID_HARTLEY_3                      1000119     Hartley 3
 NAIFID_PARKER_HARTLEY                 1000120     Parker-Hartley
 NAIFID_HELIN_ROMAN_ALU_1              1000121     Helin-Roman-Alu 1
 NAIFID_WILD_4                         1000122     Wild 4
 NAIFID_MUELLER_2                      1000123     Mueller 2
 NAIFID_MUELLER_3                      1000124     Mueller 3
 NAIFID_SHOEMAKER_LEVY_1               1000125     Shoemaker-Levy 1
 NAIFID_SHOEMAKER_LEVY_2               1000126     Shoemaker-Levy 2
 NAIFID_HOLT_OLMSTEAD                  1000127     Holt-Olmstead
 NAIFID_METCALF_BREWINGTON             1000128     Metcalf-Brewington
 NAIFID_LEVY                           1000129     Levy
 NAIFID_SHOEMAKER_LEVY_9               1000130     Shoemaker-Levy 9
 NAIFID_HYAKUTAKE                      1000131     Hyakutake
 NAIFID_HALE_BOPP                      1000132     Hale-Bopp
 NAIFID_SIDING_SPRING                  1003228     Siding Spring
===================================== ======== ====================
