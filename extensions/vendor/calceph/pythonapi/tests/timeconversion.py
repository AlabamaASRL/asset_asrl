# /*-----------------------------------------------------------------*/
# /*!
#  \file timeconversion.py
#  \brief Check if calceph_time_... works.
#
#  \author  M. Gastineau
#           Astronomie et Systemes Dynamiques, IMCCE, CNRS, Observatoire de Paris.
#
#   Copyright, 2025-2026, CNRS
#   email of the author : Mickael.Gastineau@obspm.fr
# */
# /*-----------------------------------------------------------------*/
#
# /*-----------------------------------------------------------------*/
# /* License  of this file :
# This file is "triple-licensed", you have to choose one  of the three licenses
# below to apply on this file.
#
#    CeCILL-C
#    	The CeCILL-C license is close to the GNU LGPL.
#    	( http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html )
#
# or CeCILL-B
#        The CeCILL-B license is close to the BSD.
#        (http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.txt)
#
# or CeCILL v2.1
#      The CeCILL license is compatible with the GNU GPL.
#      ( http://www.cecill.info/licences/Licence_CeCILL_V2.1-en.html )
#
#
# This library is governed by the CeCILL-C, CeCILL-B or the CeCILL license under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/ or redistribute the software under the terms
# of the CeCILL-C,CeCILL-B or CeCILL license as circulated by CEA, CNRS and INRIA
# at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C,CeCILL-B or CeCILL license and that you accept its terms.
# */
# /*-----------------------------------------------------------------*/

# /*-----------------------------------------------------------------*/
# /* main program */
# /*-----------------------------------------------------------------*/
import unittest
import openfiles
import math

from calcephpy import CalcephBin, Constants


# check the result for the calendar
def check_cal(computed_date, expected_date):
    ye, moe, de, he, mie, se = expected_date
    yc, moc, dc, hc, mic, sc = computed_date
    if (
        ye != yc
        or moe != moc
        or de != dc
        or hc != he
        or mie != mic
        or abs(se - sc) > 4e-5
    ):
        print("expected date: ", expected_date)
        print("computed date: ", computed_date)
        raise RuntimeError("invalid calendar date")


# check the result for the julianday
def check_jd(computed_date, expected_date):
    jd0e, jdfrace = expected_date
    jd0c, jdfracc = computed_date
    if abs((jd0c - jd0e) + (jdfracc - jdfrace)) > 1e-6:
        print("expected date: ", expected_date)
        print("computed date: ", computed_date)
        raise RuntimeError("invalid julian day")


class TestTime(unittest.TestCase):

    def test_time_jd_to_cal(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        computed_date = peph.time_jd_to_cal(Constants.TAI, 2457948.0, 0.76207291667)
        check_cal(computed_date, [2017, 7, 14, 6, 17, 23.1])

        computed_date = peph.time_jd_to_cal(Constants.TCB, 2457436.0, 0.11835879629)
        check_cal(computed_date, [2016, 2, 17, 14, 50, 26.2])

        computed_date = peph.time_jd_to_cal(
            Constants.UTC, 2451545.0, 0.04554398148148146
        )
        check_cal(computed_date, [2000, 1, 1, 13, 5, 35.0])

        peph.close()

    def test_time_cal_to_jd(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        computed_date = peph.time_cal_to_jd(Constants.TT, 2016, 2, 18, 14, 50, 26.2)
        check_jd(computed_date, [2457437.0, 0.11835879629])

        computed_date = peph.time_cal_to_jd(Constants.UTC, 2000, 1, 12, 13, 5, 35.0)
        check_jd(computed_date, [2451556.0, 0.04554398148148146])

        peph.close()

    def test_time_str_to_jd(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        computed_date = peph.time_str_to_jd(
            Constants.TT, "2029 OCT 16 13:14:11.4967158436775208"
        )
        check_jd(computed_date, [2462426.0, 0.0515219527296722])

        computed_date = peph.time_str_to_jd(
            Constants.TIMESCALE_FROM_STR, "1996 January 1,  06:00:0.0  (UTC)"
        )
        check_jd(computed_date, [2450083, 0.75])

        peph.close()

    def test_time_jd_tt_to_jd_tdb(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        peph.time_set_relationship_tt_tdb(1)

        computed_date = peph.time_jd_tt_to_jd_tdb(2452425, 0.989772840403)
        check_jd(computed_date, [2452425, 0.989772851113])

        computed_date = peph.time_jd_tdb_to_jd_tt(2452425, 0.989772851113)
        check_jd(computed_date, [2452425, 0.989772840403])

        peph.close()

    def test_time_jd_tcb_to_jd_tdb(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        computed_date = peph.time_jd_tcb_to_jd_tdb(2450083.0, 0.1268436964601278)
        check_jd(computed_date, [2450083.0, 0.1267361110076308])

        computed_date = peph.time_jd_tdb_to_jd_tcb(2450083.0, 0.1267361110076308)
        check_jd(computed_date, [2450083.0, 0.1268436964601278])

        peph.close()

    def test_time_str_any_to_jd_tdb(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        peph.time_set_relationship_tt_tdb(1)
        computed_date = peph.time_str_any_to_jd_tdb("1995 June 13  23:59:59.5  (UTC)")
        check_jd(computed_date, [2449882, 0.5007023680955172])

        peph.close()

    def test_time_str_utc_to_jd_tdb(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        peph.time_set_relationship_tt_tdb(1)
        computed_date = peph.time_str_utc_to_jd_tdb("1988 June 13, 12:29:48 ")
        check_jd(computed_date, [2447326, 0.0213447287678719])

        peph.close()

    def tests_time_cal_utc_to_jd_tdb(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        peph.time_set_relationship_tt_tdb(1)
        computed_date = peph.time_cal_utc_to_jd_tdb(1995, 12, 31, 23, 59, 59.5)
        check_jd(computed_date, [2450083, 0.5007023601792753])

        peph.close()

    def tests_time_jd_tdb_to_cal_utc(self):
        peph = CalcephBin.open(openfiles.prefixsrc("../../tests/example_lsk.tls"))

        peph.time_set_relationship_tt_tdb(1)
        computed_date = peph.time_jd_tdb_to_cal_utc(2450083, 0.5007023601792753)
        check_cal(computed_date, [1995, 12, 31, 23, 59, 59.5])

        computed_date = peph.time_jd_tdb_to_cal_utc(
            2450083, 0.5007023601792753 + 1.0 / 86401.0
        )
        check_cal(computed_date, [1995, 12, 31, 23, 59, 60.5])

        peph.close()

    def tests_time_str_spacecraft_clock_to_jd_tdb(self):
        peph = CalcephBin.open([openfiles.prefixsrc("../../tests/example_lsk.tls"), openfiles.prefixsrc("../../tests/example_sclk.tsc")])

        peph.time_set_relationship_tt_tdb(1)
        computed_date = peph.time_str_spacecraft_clock_to_jd_tdb(28, "1/0706865508:31865")
        check_jd(computed_date, [2459725, 0.81458599643883644603 ])

        computed_date = peph.time_jd_tdb_to_str_spacecraft_clock(
            28,  2459744,  0.35689482379893888719
        )
        expected_date = "1/0708467563:63527"
        if (computed_date!=expected_date):
            print("computed date", computed_date)
            print("expected date", expected_date)
            raise RuntimeError("time_jd_tdb_to_str_spacecraft_clock fails")
        
        peph.close()


if __name__ == "__main__":
    unittest.main()
