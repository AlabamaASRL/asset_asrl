% NAIF identification numbers of the CALCEPH library.

%  /*-----------------------------------------------------------------*/
%  /*! 
%    \file NaifId.m
%    \brief MEX interface for the class NaifId
%  
%    \author  M. Gastineau 
%             Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris. 
%  
%     Copyright,  2018, CNRS
%     email of the author : Mickael.Gastineau@obspm.fr
%  
%    History:
%  */
%  /*-----------------------------------------------------------------*/
%  
%   /*-----------------------------------------------------------------*/
%   /* License  of this file :
%    This file is "triple-licensed", you have to choose one  of the three licenses 
%    below to apply on this file.
%    
%       CeCILL-C
%       	The CeCILL-C license is close to the GNU LGPL.
%       	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
%      
%    or CeCILL-B
%           The CeCILL-B license is close to the BSD.
%           (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
%     
%    or CeCILL v2.1;
%         The CeCILL license is compatible with the GNU GPL.
%         ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
%    
%   
%   This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under 
%   French law and abiding by the rules of distribution of free software.  
%   You can  use, modify and/ or redistribute the software under the terms 
%   of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA  
%   at the following URL "http://www.cecill.info". 
%   
%   As a counterpart to the access to the source code and  rights to copy,
%   modify and redistribute granted by the license, users are provided only
%   with a limited warranty  and the software's author,  the holder of the
%   economic rights,  and the successive licensors  have only  limited
%   liability. 
%   
%   In this respect, the user's attention is drawn to the risks associated
%   with loading,  using,  modifying and/or developing or reproducing the
%   software by the user in light of its specific status of free software,
%   that may mean  that it is complicated to manipulate,  and  that  also
%   therefore means  that it is reserved for developers  and  experienced
%   professionals having in-depth computer knowledge. Users are therefore
%   encouraged to load and test the software's suitability as regards their
%   requirements in conditions enabling the security of their systems and/or 
%   data to be ensured and,  more generally, to use and operate it in the 
%   same conditions as regards security. 
%   
%   The fact that you are presently reading this means that you have had
%   knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
%   */
%   /*-----------------------------------------------------------------*/

classdef NaifId

   properties (GetAccess = public, Constant = true)
%/*! NAIF identification numbers for the Sun and planetary barycenters (table 2 of reference 1) */
    SOLAR_SYSTEM_BARYCENTER  = 0;     
    MERCURY_BARYCENTER       = 1;
    VENUS_BARYCENTER         = 2;
    EARTH_MOON_BARYCENTER    = 3;
    MARS_BARYCENTER          = 4;
    JUPITER_BARYCENTER       = 5;
    SATURN_BARYCENTER        = 6;
    URANUS_BARYCENTER        = 7;
    NEPTUNE_BARYCENTER       = 8;
    PLUTO_BARYCENTER         = 9;
    SUN                      = 10;

%/*! NAIF identification numbers for the Coordinate Time ephemerides */
%/*! value to set as the center to get any Coordinate Time */
    TIME_CENTER   = 1000000000;        
%/*! value to set as the target to get the Coordinate Time TT-TDB */
    TIME_TTMTDB   = 1000000001;
%/*! value to set as the target to get the Coordinate Time TCG-TCB */
    TIME_TCGMTCB  = 1000000002;

%/*! NAIF identification numbers for the planet centers and satellites (table= 3 of reference= 1)  */
    MERCURY       = 199;
    VENUS         = 299;
    EARTH         = 399;
    MOON          = 301;
  
    MARS          = 499;
    PHOBOS        = 401;
    DEIMOS        = 402;
    
    JUPITER       = 599;
    IO            = 501;
    EUROPA        = 502;
    GANYMEDE      = 503;
    CALLISTO      = 504;
    AMALTHEA      = 505;
    HIMALIA       = 506;
    ELARA         = 507;
    PASIPHAE      = 508;
    SINOPE        = 509;
    LYSITHEA      = 510;
    CARME         = 511;
    ANANKE        = 512;
    LEDA          = 513;
    THEBE         = 514;
    ADRASTEA      = 515;
    METIS         = 516;
    CALLIRRHOE    = 517;
    THEMISTO      = 518;
    MEGACLITE     = 519;
    TAYGETE       = 520;
    CHALDENE      = 521;
    HARPALYKE     = 522;
    KALYKE        = 523;
    IOCASTE       = 524;
    ERINOME       = 525;
    ISONOE        = 526;
    PRAXIDIKE     = 527;
    AUTONOE       = 528;
    THYONE        = 529;
    HERMIPPE      = 530;
    AITNE         = 531;
    EURYDOME      = 532;
    EUANTHE       = 533;
    EUPORIE       = 534;
    ORTHOSIE      = 535;
    SPONDE        = 536;
    KALE          = 537;
    PASITHEE      = 538;
    HEGEMONE      = 539;
    MNEME         = 540;
    AOEDE         = 541;
    THELXINOE     = 542;
    ARCHE         = 543;
    KALLICHORE    = 544;
    HELIKE        = 545;
    CARPO         = 546;
    EUKELADE      = 547;
    CYLLENE       = 548;
    KORE          = 549;
    HERSE         = 550;
    DIA           = 553;
 
    SATURN        = 699;
    MIMAS         = 601;
    ENCELADUS     = 602;
    TETHYS        = 603;
    DIONE         = 604;
    RHEA          = 605;
    TITAN         = 606;
    HYPERION      = 607;
    IAPETUS       = 608;
    PHOEBE        = 609;
    JANUS         = 610;
    EPIMETHEUS    = 611;
    HELENE        = 612;
    TELESTO       = 613;
    CALYPSO       = 614;
    ATLAS         = 615;
    PROMETHEUS    = 616;
    PANDORA       = 617;
    PAN           = 618;
    YMIR          = 619;
    PAALIAQ       = 620;
    TARVOS        = 621;
    IJIRAQ        = 622;
    SUTTUNGR      = 623;
    KIVIUQ        = 624;
    MUNDILFARI    = 625;
    ALBIORIX      = 626;
    SKATHI        = 627;
    ERRIAPUS      = 628;
    SIARNAQ       = 629;
    THRYMR        = 630;
    NARVI         = 631;
    METHONE       = 632;
    PALLENE       = 633;
    POLYDEUCES    = 634;
    DAPHNIS       = 635;
    AEGIR         = 636;
    BEBHIONN      = 637;
    BERGELMIR     = 638;
    BESTLA        = 639;
    FARBAUTI      = 640;
    FENRIR        = 641;
    FORNJOT       = 642;
    HATI          = 643;
    HYROKKIN      = 644;
    KARI          = 645;
    LOGE          = 646;
    SKOLL         = 647;
    SURTUR        = 648;
    ANTHE         = 649;
    JARNSAXA      = 650;
    GREIP         = 651;
    TARQEQ        = 652;
    AEGAEON       = 653;

    URANUS        = 799;
    ARIEL         = 701;
    UMBRIEL       = 702;
    TITANIA       = 703;
    OBERON        = 704;
    MIRANDA       = 705;
    CORDELIA      = 706;
    OPHELIA       = 707;
    BIANCA        = 708;
    CRESSIDA      = 709;
    DESDEMONA     = 710;
    JULIET        = 711;
    PORTIA        = 712;
    ROSALIND      = 713;
    BELINDA       = 714;
    PUCK          = 715;
    CALIBAN       = 716;
    SYCORAX       = 717;
    PROSPERO      = 718;
    SETEBOS       = 719;
    STEPHANO      = 720;
    TRINCULO      = 721;
    FRANCISCO     = 722;
    MARGARET      = 723;
    FERDINAND     = 724;
    PERDITA       = 725;
    MAB           = 726;
    CUPID         = 727;
 
    NEPTUNE       = 899;
    TRITON        = 801;
    NEREID        = 802;
    NAIAD         = 803;
    THALASSA      = 804;
    DESPINA       = 805;
    GALATEA       = 806;
    LARISSA       = 807;
    PROTEUS       = 808;
    HALIMEDE      = 809;
    PSAMATHE      = 810;
    SAO           = 811;
    LAOMEDEIA     = 812;
    NESO          = 813;

    PLUTO         = 999;
    CHARON        = 901;
    NIX           = 902;
    HYDRA         = 903;
    KERBEROS      = 904;
    STYX          = 905;

%/*! NAIF identification numbers for the comets (table= 4 of reference= 1)  */
    AREND                              = 1000001;
    AREND_RIGAUX                       = 1000002;
    ASHBROOK_JACKSON                   = 1000003;
    BOETHIN                            = 1000004;
    BORRELLY                           = 1000005;
    BOWELL_SKIFF                       = 1000006;
    BRADFIELD                          = 1000007;
    BROOKS_2                           = 1000008;
    BRORSEN_METCALF                    = 1000009;
    BUS                                = 1000010;
    CHERNYKH                           = 1000011;
    CHURYUMOV_GERASIMENKO              = 1000012;
    CIFFREO                            = 1000013;
    CLARK                              = 1000014;
    COMAS_SOLA                         = 1000015;
    CROMMELIN                          = 1000016;
    D__ARREST                          = 1000017;
    DANIEL                             = 1000018;
    DE_VICO_SWIFT                      = 1000019;
    DENNING_FUJIKAWA                   = 1000020;
    DU_TOIT_1                          = 1000021;
    DU_TOIT_HARTLEY                    = 1000022;
    DUTOIT_NEUJMIN_DELPORTE            = 1000023;
    DUBIAGO                            = 1000024;
    ENCKE                              = 1000025;
    FAYE                               = 1000026;
    FINLAY                             = 1000027;
    FORBES                             = 1000028;
    GEHRELS_1                          = 1000029;
    GEHRELS_2                          = 1000030;
    GEHRELS_3                          = 1000031;
    GIACOBINI_ZINNER                   = 1000032;
    GICLAS                             = 1000033;
    GRIGG_SKJELLERUP                   = 1000034;
    GUNN                               = 1000035;
    HALLEY                             = 1000036;
    HANEDA_CAMPOS                      = 1000037;
    HARRINGTON                         = 1000038;
    HARRINGTON_ABELL                   = 1000039;
    HARTLEY_1                          = 1000040;
    HARTLEY_2                          = 1000041;
    HARTLEY_IRAS                       = 1000042;
    HERSCHEL_RIGOLLET                  = 1000043;
    HOLMES                             = 1000044;
    HONDA_MRKOS_PAJDUSAKOVA            = 1000045;
    HOWELL                             = 1000046;
    IRAS                               = 1000047;
    JACKSON_NEUJMIN                    = 1000048;
    JOHNSON                            = 1000049;
    KEARNS_KWEE                        = 1000050;
    KLEMOLA                            = 1000051;
    KOHOUTEK                           = 1000052;
    KOJIMA                             = 1000053;
    KOPFF                              = 1000054;
    KOWAL_1                            = 1000055;
    KOWAL_2                            = 1000056;
    KOWAL_MRKOS                        = 1000057;
    KOWAL_VAVROVA                      = 1000058;
    LONGMORE                           = 1000059;
    LOVAS_1                            = 1000060;
    MACHHOLZ                           = 1000061;
    MAURY                              = 1000062;
    NEUJMIN_1                          = 1000063;
    NEUJMIN_2                          = 1000064;
    NEUJMIN_3                          = 1000065;
    OLBERS                             = 1000066;
    PETERS_HARTLEY                     = 1000067;
    PONS_BROOKS                        = 1000068;
    PONS_WINNECKE                      = 1000069;
    REINMUTH_1                         = 1000070;
    REINMUTH_2                         = 1000071;
    RUSSELL_1                          = 1000072;
    RUSSELL_2                          = 1000073;
    RUSSELL_3                          = 1000074;
    RUSSELL_4                          = 1000075;
    SANGUIN                            = 1000076;
    SCHAUMASSE                         = 1000077;
    SCHUSTER                           = 1000078;
    SCHWASSMANN_WACHMANN_1             = 1000079;
    SCHWASSMANN_WACHMANN_2             = 1000080;
    SCHWASSMANN_WACHMANN_3             = 1000081;
    SHAJN_SCHALDACH                    = 1000082;
    SHOEMAKER_1                        = 1000083;
    SHOEMAKER_2                        = 1000084;
    SHOEMAKER_3                        = 1000085;
    SINGER_BREWSTER                    = 1000086;
    SLAUGHTER_BURNHAM                  = 1000087;
    SMIRNOVA_CHERNYKH                  = 1000088;
    STEPHAN_OTERMA                     = 1000089;
    SWIFT_GEHRELS                      = 1000090;
    TAKAMIZAWA                         = 1000091;
    TAYLOR                             = 1000092;
    TEMPEL_1                           = 1000093;
    TEMPEL_2                           = 1000094;
    TEMPEL_TUTTLE                      = 1000095;
    TRITTON                            = 1000096;
    TSUCHINSHAN_1                      = 1000097;
    TSUCHINSHAN_2                      = 1000098;
    TUTTLE                             = 1000099;
    TUTTLE_GIACOBINI_KRESAK            = 1000100;
    VAISALA_1                          = 1000101;
    VAN_BIESBROECK                     = 1000102;
    VAN_HOUTEN                         = 1000103;
    WEST_KOHOUTEK_IKEMURA              = 1000104;
    WHIPPLE                            = 1000105;
    WILD_1                             = 1000106;
    WILD_2                             = 1000107;
    WILD_3                             = 1000108;
    WIRTANEN                           = 1000109;
    WOLF                               = 1000110;
    WOLF_HARRINGTON                    = 1000111;
    LOVAS_2                            = 1000112;
    URATA_NIIJIMA                      = 1000113;
    WISEMAN_SKIFF                      = 1000114;
    HELIN                              = 1000115;
    MUELLER                            = 1000116;
    SHOEMAKER_HOLT_1                   = 1000117;
    HELIN_ROMAN_CROCKETT               = 1000118;
    HARTLEY_3                          = 1000119;
    PARKER_HARTLEY                     = 1000120;
    HELIN_ROMAN_ALU_1                  = 1000121;
    WILD_4                             = 1000122;
    MUELLER_2                          = 1000123;
    MUELLER_3                          = 1000124;
    SHOEMAKER_LEVY_1                   = 1000125;
    SHOEMAKER_LEVY_2                   = 1000126;
    HOLT_OLMSTEAD                      = 1000127;
    METCALF_BREWINGTON                 = 1000128;
    LEVY                               = 1000129;
    SHOEMAKER_LEVY_9                   = 1000130;
    HYAKUTAKE                          = 1000131;
    HALE_BOPP                          = 1000132;
    SIDING_SPRING                      = 1003228;
    end
end    