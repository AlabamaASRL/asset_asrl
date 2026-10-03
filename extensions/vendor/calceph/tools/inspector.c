/*-----------------------------------------------------------------*/
/*!
  \file inspector.c
  \brief display details about the ephemeris file.

  \author  M. Gastineau
           Astronomie et Systemes Dynamiques, LTE, CNRS, Observatoire de
  Paris.

   Copyright, 2011-2026, CNRS
   email of the author : Mickael.Gastineau@obspm.fr

  History:
*/
/*-----------------------------------------------------------------*/

/*-----------------------------------------------------------------*/
/* License  of this file :
 This file is "triple-licensed", you have to choose one  of the three licenses
 below to apply on this file.

    CeCILL-C
        The CeCILL-C license is close to the GNU LGPL.
        ( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )

 or CeCILL-B
        The CeCILL-B license is close to the BSD.
        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)

 or CeCILL v2.1
      The CeCILL license is compatible with the GNU GPL.
      ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )

This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under
French law and abiding by the rules of distribution of free software.
You can  use, modify and/ or redistribute the software under the terms
of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA
at the following URL "http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its
terms.
*/
/*-----------------------------------------------------------------*/

#include "calcephconfig.h"
#if HAVE_STDIO_H
#include <stdio.h>
#endif
#if HAVE_STDLIB_H
#include <stdlib.h>
#endif
#if HAVE_STRING_H
#include <string.h>
#endif
#if HAVE_ERRNO_H
#include <errno.h>
#endif
#if HAVE_MATH_H
#include <math.h>
#endif
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifndef __cplusplus
#if HAVE_STDBOOL_H
#include <stdbool.h>
#endif
#endif /*__cplusplus */

#include "real.h"
#include "calceph.h"
#include "calcephinternal.h"
#include "util.h"

static void print_txt_pck(struct SPICEkernel *pkernel, const char* title)
{
    struct TXTPCKfile *ptxtpck;

    struct TXTPCKconstant *listconstant;

    printf("%s\n", title);
    ptxtpck = &pkernel->filedata.txtpck;
    printf("list of constant of the %s : \n", title);
    for (listconstant = ptxtpck->listconstant; listconstant != NULL;
            listconstant = listconstant->next)
    {
        struct TXTPCKvalue *listvalue;

        printf("'%s' = ( ", listconstant->name);
        for (listvalue = listconstant->value; listvalue != NULL; listvalue = listvalue->next)
        {
            printf("%.*s", (int) (listvalue->loclast - listvalue->locfirst),
                    ptxtpck->buffer + listvalue->locfirst);
            if (listvalue->next != NULL)
                printf(" , ");
        }
        printf(" ) \n");
    }
}


int main(int argc, char *argv[])
{
    t_calcephbin *peph;

    const char *filename;

    struct calcephbin_inpop *pinpop;

    struct calcephbin_spice *pspice;

    struct SPICEkernel *pkernel;

    struct SPKfile *pspk;

    struct TXTPCKconstant *listconstant;

    struct TXTFKfile *ptxtfk;

    struct TXTFKframe *listframe;

    struct SPKSegmentList *listseg;

    char nameconstant[CALCEPH_MAX_CONSTANTNAME];

    double valueconstant;

    char fileversion[CALCEPH_MAX_CONSTANTVALUE];

    int j;

    int timescale;

    double firsttime, lasttime;

    int continuous;

    int countposrecord;

    int countorirecord;

    int target, center, frame;

    if (argc != 2)
    {
        printf("Usage : calceph_inspector filename\n");
        printf("                          filename : binary ephemeris file name\n\n");
        printf("e.g.  : calceph_inspector inpop08a_m100_p100_littleendian.dat\n\n");
        return 1;
    }
    printf("calceph_inspector -  CALCEPH version %d.%d.%d\n\n", CALCEPH_VERSION_MAJOR,
           CALCEPH_VERSION_MINOR, CALCEPH_VERSION_PATCH);
    filename = argv[1];
    peph = calceph_open(filename);
    if (peph)
    {
        if (calceph_getfileversion(peph, fileversion))
        {
            printf("Ephemeris file version : '%s'\n", fileversion);
        }
        else
        {
            printf("Ephemeris file version : unknown\n");
        }

        timescale = calceph_gettimescale(peph);
        printf("timescale : ");
        switch (timescale)
        {
            case 1:
                printf("TDB\n");
                break;
            case 2:
                printf("TCB\n");
                break;
            default:
                printf("unknown\n");
                break;
        }

        countposrecord = calceph_getpositionrecordcount(peph);
        printf("Number of position's record : %d\n", countposrecord);
        countorirecord = calceph_getorientrecordcount(peph);
        printf("Number of orientation's record : %d\n", countorirecord);

        switch (peph->etype)
        {
            case CALCEPH_espice:
                printf("SPICE file format\n");
                pspice = &peph->data.spkernel;
                /*if (pspice==NULL) printf("peph->data.spkernel is NULL\n"); */
                pkernel = pspice->list;
                while (pkernel != NULL)
                {
                    switch (pkernel->filetype)
                    {
                        case DAF_SPK:
                            printf("SPK file\n");
                            pspk = &pkernel->filedata.spk;
                            printf("Binary file format ID : '%.*s'\n", IFNAME_LEN, pspk->header.bff);
                            printf("body        center      start time (JD) end time (JD)     "
                                   "frame  data type    data information\n");
                            printf("-------------------------------------------------------------"
                                   "-----------------------------------------\n");
                            for (listseg = pspk->list_seg; listseg != NULL; listseg = listseg->next)
                            {
                                for (j = 0; j < listseg->array_seg.count; j++)
                                {
                                    struct SPKSegmentHeader *seg = listseg->array_seg.array + j;

                                    printf("%-12d %-12d%-16.2f%-16.2f %-10d%-12d", seg->body, seg->center,
                                           seg->T_begin / 86400E0 + 2.451545E+06, seg->T_end / 86400E0 + 2.451545E+06,
                                           seg->refframe, seg->datatype);
                                    switch (seg->datatype)
                                    {
                                        case SPK_SEGTYPE1:
                                            printf("MDA number of records : %6d  TDB\n",
                                                   seg->seginfo.data1.count_record);
                                            break;

                                        case SPK_SEGTYPE3:
                                        case SPK_SEGTYPE2:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TDB\n",
                                                   seg->seginfo.data2.T_len / 86400E0, seg->seginfo.data2.count_record);
                                            break;

                                        case SPK_SEGTYPE103:
                                        case SPK_SEGTYPE102:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TCB\n",
                                                   seg->seginfo.data2.T_len / 86400E0, seg->seginfo.data2.count_record);
                                            break;

                                        case SPK_SEGTYPE5:
                                            printf
                                                ("Discrete states (two body propagation) number of records : %6d GM: %g  "
                                                 "TDB\n", seg->seginfo.data5.count_record, seg->seginfo.data5.GM);
                                            break;

                                        case SPK_SEGTYPE8:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d\tLagrange Interpolation of degree %d - Equal Time "
                                                   "Steps  TDB\n",
                                                   seg->seginfo.data12.step_size / 86400E0,
                                                   seg->seginfo.data12.count_record, seg->seginfo.data12.degree);
                                            break;

                                        case SPK_SEGTYPE9:
                                            printf("number of records : %6d\tLagrange Interpolation of "
                                                   "degree %d - Unequal Time Steps  TDB\n",
                                                   seg->seginfo.data13.count_record, seg->seginfo.data13.degree);
                                            break;

                                        case SPK_SEGTYPE12:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d\tHermite Interpolation of degree %d - Equal Time "
                                                   "Steps  TDB\n",
                                                   seg->seginfo.data12.step_size / 86400E0,
                                                   seg->seginfo.data12.count_record, seg->seginfo.data12.degree);
                                            break;

                                        case SPK_SEGTYPE13:
                                            printf("number of records : %6d\tHermite Interpolation of "
                                                   "degree %d - Unequal Time Steps  TDB\n",
                                                   seg->seginfo.data13.count_record, seg->seginfo.data13.degree);
                                            break;

                                        case SPK_SEGTYPE14:
                                            printf("number of packets : %6d\tChebychev of "
                                                   "degree %d - Unequal Time Steps  TDB\n",
                                                   seg->seginfo.data14.count_packets, seg->seginfo.data14.degree);
                                            break;
                                            
                                        case SPK_SEGTYPE17:
                                            printf("Equinoctial Elements : epoch=%g TDB\n", seg->seginfo.data17.epoch);
                                            break;

                                        case SPK_SEGTYPE18:
                                            printf("number of records : %6d\tESOC/DDID %s Interpolation of "
                                                   "degree %d - Window's size %d - Unequal Time Steps  TDB\n",
                                                   seg->seginfo.data18.count_record,
                                                   seg->seginfo.data18.subtype == 0 ? "Hermite" : "Lagrange",
                                                   seg->seginfo.data18.degree, seg->seginfo.data18.window_size);
                                            break;

                                        case SPK_SEGTYPE19:
                                            printf("number of records : %6d\tESOC/DDID Piecewise Interpolation "
                                                   "- boundary flag %d -  TDB\n",
                                                   seg->seginfo.data19.number_intervals,
                                                   seg->seginfo.data19.boundary_choice_flag);
                                            break;

                                        case SPK_SEGTYPE20:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TDB Dscale=%g Tscale=%g\n",
                                                   seg->seginfo.data20.T_len / 86400E0,
                                                   seg->seginfo.data20.count_record, seg->seginfo.data20.dscale,
                                                   seg->seginfo.data20.tscale);
                                            break;

                                        case SPK_SEGTYPE21:
                                            printf("Extended MDA number of records : %6d dlsize=%6d TDB\n",
                                                   seg->seginfo.data21.count_record, seg->seginfo.data21.dlsize);
                                            break;

                                        case SPK_SEGTYPE120:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TCB Dscale=%g Tscale=%g\n",
                                                   seg->seginfo.data20.T_len / 86400E0,
                                                   seg->seginfo.data20.count_record, seg->seginfo.data20.dscale,
                                                   seg->seginfo.data20.tscale);
                                            break;

                                        default:
                                            printf("Unsupported segment (type=%d).", (int) seg->datatype);
                                            break;
                                    }
                                }
                                printf("\n");
                            }
                            break;

                        case TXT_PCK:
                            print_txt_pck(pkernel, "text PCK file");
                            break;

                        case TXT_IK:
                            print_txt_pck(pkernel, "instrument file");
                            break;

                        case TXT_LSK:
                            print_txt_pck(pkernel, "leapsecond file");
                            break;
                            
                        case TXT_SCLK:
                            print_txt_pck(pkernel, "spacecraft clock file");
                            break;
 
                        case DAF_PCK:
                            printf("binary PCK file\n");
                            pspk = &pkernel->filedata.spk;
                            printf("Binary file format ID : '%.*s'\n", IFNAME_LEN, pspk->header.bff);
                            printf("body     start time (JD)  end time (JD)    frame  data type  "
                                   " data information\n");
                            printf("-------------------------------------------------------------"
                                   "------------------------\n");
                            for (listseg = pspk->list_seg; listseg != NULL; listseg = listseg->next)
                            {
                                for (j = 0; j < listseg->array_seg.count; j++)
                                {
                                    struct SPKSegmentHeader *seg = listseg->array_seg.array + j;

                                    printf("%-8d %-16.2f %-16.2f %-7d%-12d", seg->body,
                                           seg->T_begin / 86400E0 + 2.451545E+06, seg->T_end / 86400E0 + 2.451545E+06,
                                           seg->refframe, seg->datatype);
                                    switch (seg->datatype)
                                    {
                                        case SPK_SEGTYPE3:
                                        case SPK_SEGTYPE2:
                                            printf("Time span per record: %4.2g (day)  number of records : "
                                                   "%6d  TDB\n",
                                                   seg->seginfo.data2.T_len / 86400E0, seg->seginfo.data2.count_record);
                                            break;

                                        case SPK_SEGTYPE103:
                                        case SPK_SEGTYPE102:
                                            printf("Time span per record: %4.2g (day)  number of records : "
                                                   "%6d  TCB\n",
                                                   seg->seginfo.data2.T_len / 86400E0, seg->seginfo.data2.count_record);
                                            break;

                                        case SPK_SEGTYPE20:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TDB Dscale=%g Tscale=%g\n",
                                                   seg->seginfo.data20.T_len /*/ 86400E0 */ ,
                                                   seg->seginfo.data20.count_record,
                                                   seg->seginfo.data20.dscale, seg->seginfo.data20.tscale);
                                            break;

                                        case SPK_SEGTYPE120:
                                            printf("Time span per record: %4.2g (day)\tnumber of records : "
                                                   "%6d  TCB Dscale=%g Tscale=%g\n",
                                                   seg->seginfo.data20.T_len /*/ 86400E0 */ ,
                                                   seg->seginfo.data20.count_record,
                                                   seg->seginfo.data20.dscale, seg->seginfo.data20.tscale);
                                            break;

                                        default:
                                            printf("Unsupported segment (type=%d).", (int) seg->datatype);
                                            break;
                                    }
                                }
                                printf("\n");
                            }
                            break;

                        case TXT_FK:
                            printf("text FK file\n");
                            ptxtfk = &pkernel->filedata.txtfk;
                            printf("list of frame of the text FK file : \n");
                            for (listframe = ptxtfk->listframe; listframe != NULL; listframe = listframe->next)
                            {
                                printf("'%s' = ( %d, class=%2d, classid=%8d, center = %8d ", listframe->name,
                                       listframe->frame_id, listframe->classtype, listframe->class_id,
                                       listframe->center);

                                printf(" ) \n");
                            }
                            printf("list of constant of the text FK file : \n");
                            for (listconstant = ptxtfk->txtpckfile.listconstant; listconstant != NULL;
                                 listconstant = listconstant->next)
                            {
                                struct TXTPCKvalue *listvalue;

                                printf("'%s' = ( ", listconstant->name);
                                for (listvalue = listconstant->value; listvalue != NULL; listvalue = listvalue->next)
                                {
                                    printf("%.*s", (int) (listvalue->loclast - listvalue->locfirst),
                                           ptxtfk->txtpckfile.buffer + listvalue->locfirst);
                                    if (listvalue->next != NULL)
                                        printf(" , ");
                                }
                                printf(" ) \n");
                            }
                            break;

                        default:
                            printf("Unknown SPICE type in %d\n", (int) pkernel->filetype);
							return 2;
                            break;
                    }
                    pkernel = pkernel->next;
                }
                break;

            case CALCEPH_ebinary:
                printf("INPOP or original DE file format\n");
                pinpop = &peph->data.binary;
                printf("Ephemerides number : %d\n", pinpop->H1.DENUM);
                calceph_gettimespan(peph, &firsttime, &lasttime, &continuous);
                printf("Start time : %23.16e\n", firsttime);
                printf("End time : %23.16e\n", lasttime);
                printf("Time Span : %23.16e\n", pinpop->H1.timeData[2]);
                printf("Data available on continuous time : %d\n", continuous);
                printf("1980 IAU nutation angles : %s\n", pinpop->haveNutation ? "available" : "not available");
                printf("TT-TDB                   : %s\n", pinpop->haveTTmTDB ? "available" : "not available");
                printf("Lunar angular velocity   : %s\n",
                       pinpop->haveLunarAngularVelocity ? "available" : "not available");

                printf("\nNumber of constants : %d\n", calceph_getconstantcount(peph));
                printf("list of constants :\n");
                for (j = 1; j <= calceph_getconstantcount(peph); j++)
                {
                    calceph_getconstantindex(peph, j, nameconstant, &valueconstant);
                    printf("'%s'\t= %23.16E\n", nameconstant, valueconstant);
                }

                if (countposrecord)
                {
                    printf("position's records :\n");
                    printf("body        center      start time (JD)         end time (JD)  " "   frame  \n");
                    printf("---------------------------------------------------------------"
                           "----------------------\n");
                    for (j = 1; j <= countposrecord; j++)
                    {
                        calceph_getpositionrecordindex(peph, j, &target, &center, &firsttime, &lasttime, &frame);
                        printf("%8d %8d %23.16E %23.16E %4d\n", target, center, firsttime, lasttime, frame);
                    }
                }
                if (countorirecord)
                {
                    printf("orientation's records :\n");
                    printf("body          start time (JD)         end time (JD)     frame  " "\n");
                    printf("---------------------------------------------------------------"
                           "----------------------\n");
                    for (j = 1; j <= countorirecord; j++)
                    {
                        calceph_getorientrecordindex(peph, j, &target, &firsttime, &lasttime, &frame);
                        printf("%8d %23.16E %23.16E %4d\n", target, firsttime, lasttime, frame);
                    }
                }
                break;

            default:
                printf("Unknown ephemeris type in calceph_getconstant\n");
				return 1;
                break;
        }

        calceph_close(peph);
    }
    else return 1;
    return 0;
}
