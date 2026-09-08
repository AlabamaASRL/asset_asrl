"""Physics-facing regression tests for astronomy units and Julian dates."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


# Locate implementation modules relative to the installed test layout.

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATE_MODULE = PACKAGE_ROOT / "Astro" / "Date.py"
CONSTANTS_MODULE = PACKAGE_ROOT / "Astro" / "Constants.py"


def load_module(name: str, path: pathlib.Path):
    """Load a self-contained module directly."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


date = load_module("asset_asrl_date", DATE_MODULE)
constants = load_module("asset_asrl_constants", CONSTANTS_MODULE)


class JulianDateTests(unittest.TestCase):

    def test_reference_epoch_j2000(self):
        """J2000 must correspond to JD 2451545.0."""
        self.assertEqual(
            date.date_to_jd(2000, 1, 1.5),
            2451545.0,
        )

        self.assertEqual(
            date.JD_SPJ2000D(2451545.0),
            0.0,
        )

    def test_known_astronomical_example(self):
        """Verify the Duffet-Smith reference example."""
        self.assertEqual(
            date.date_to_jd(1985, 2, 17.25),
            2446113.75,
        )

        self.assertEqual(
            date.jd_to_date(2446113.75),
            (1985, 2, 17.25),
        )

    def test_mjd_and_jd_are_inverse_conversions(self):
        """MJD and JD conversions must exactly invert each other."""
        for jd in (
            0.0,
            2400000.5,
            2451545.0,
            2460000.25,
            -1000.5,
        ):
            self.assertEqual(
                date.mjd_to_jd(date.jd_to_mjd(jd)),
                jd,
            )

    def test_jd_to_mjd_offset_is_exact(self):
        """JD and MJD must differ by exactly 2400000.5 days."""
        for jd in (
            0.0,
            2451545.0,
            2460000.5,
        ):
            self.assertEqual(
                date.jd_to_mjd(jd),
                jd - 2400000.5,
            )

    def test_fractional_day_conversion(self):
        """Verify common fractions of a day."""
        test_cases = (
            (0.0, (0, 0, 0, 0)),
            (0.25, (6, 0, 0, 0)),
            (0.5, (12, 0, 0, 0)),
            (0.75, (18, 0, 0, 0)),
        )

        for fraction, expected in test_cases:
            self.assertEqual(
                date.days_to_hmsm(fraction),
                expected,
            )

    def test_hmsm_conversion_at_midnight(self):
        """Midnight must correspond to zero fractional days."""
        self.assertEqual(
            date.hmsm_to_days(
                hour=0,
                mins=0,
                sec=0,
                micro=0,
            ),
            0.0,
        )

    def test_hmsm_conversion_at_end_of_day(self):
        """23:59:59.999999 must remain inside the same day."""
        fraction = date.hmsm_to_days(
            hour=23,
            mins=59,
            sec=59,
            micro=999999,
        )

        self.assertLess(fraction, 1.0)

        self.assertEqual(
            date.days_to_hmsm(fraction),
            (23, 59, 59, 999999),
        )

    def test_hmsm_and_fractional_day_are_inverse(self):
        """Time-of-day conversion should round-trip."""
        test_cases = (
            (0, 0, 0, 0),
            (1, 2, 3, 4),
            (6, 30, 15, 500000),
            (12, 0, 0, 1),
            (23, 59, 59, 999999),
        )

        for hour, minute, second, microsecond in test_cases:
            fraction = date.hmsm_to_days(
                hour,
                minute,
                second,
                microsecond,
            )

            recovered = date.days_to_hmsm(fraction)

            self.assertEqual(
                recovered,
                (
                    hour,
                    minute,
                    second,
                    microsecond,
                ),
            )

    def test_days_to_hmsm_rejects_negative_days(self):
        """Negative fractional days are invalid."""
        with self.assertRaises(ValueError):
            date.days_to_hmsm(-1.0)

    def test_days_to_hmsm_rejects_one_day(self):
        """Exactly one day must be rejected because it is not fractional."""
        with self.assertRaises(ValueError):
            date.days_to_hmsm(1.0)

    def test_days_to_hmsm_rejects_values_greater_than_one(self):
        """Values greater than one day are invalid."""
        with self.assertRaises(ValueError):
            date.days_to_hmsm(1.5)

    def test_calendar_datetime_round_trip_preserves_microseconds(self):
        """Datetime -> JD -> datetime should preserve time to float precision."""
        original = date.datetime(
            2024,
            2,
            29,
            12,
            34,
            56,
            123456,
        )

        recovered = date.jd_to_datetime(
            date.datetime_to_jd(original)
        )

        self.assertLessEqual(
            abs(recovered - original),
            date.dt.timedelta(microseconds=25),
        )

    def test_datetime_to_jd_midnight(self):
        """Midnight should correspond to an integer-plus-half JD."""
        original = date.datetime(
            2000,
            1,
            1,
            0,
            0,
            0,
        )

        self.assertEqual(
            date.datetime_to_jd(original),
            2451544.5,
        )

    def test_datetime_to_jd_j2000(self):
        """J2000 occurs at noon on January 1, 2000."""
        original = date.datetime(
            2000,
            1,
            1,
            12,
            0,
            0,
        )

        self.assertEqual(
            date.datetime_to_jd(original),
            2451545.0,
        )

    def test_jd_to_datetime_j2000(self):
        """JD 2451545.0 must convert to J2000 noon."""
        recovered = date.jd_to_datetime(2451545.0)

        self.assertEqual(
            recovered,
            date.datetime(
                2000,
                1,
                1,
                12,
                0,
                0,
            ),
        )

    def test_datetime_round_trip_across_leap_day(self):
        """Leap-day dates must survive JD conversion."""
        original = date.datetime(
            2024,
            2,
            29,
            23,
            59,
            59,
            123456,
        )

        recovered = date.jd_to_datetime(
            date.datetime_to_jd(original)
        )

        self.assertLessEqual(
            abs(recovered - original),
            date.dt.timedelta(microseconds=25),
        )

    def test_datetime_round_trip_across_year_boundary(self):
        """Dates crossing New Year's Eve must round-trip correctly."""
        original = date.datetime(
            2023,
            12,
            31,
            23,
            59,
            59,
            999999,
        )

        recovered = date.jd_to_datetime(
            date.datetime_to_jd(original)
        )

        self.assertLessEqual(
            abs(recovered - original),
            date.dt.timedelta(microseconds=25),
        )

    def test_timedelta_conversion_preserves_microseconds(self):
        """Timedelta conversion should preserve fractional seconds."""
        duration = date.dt.timedelta(
            days=2,
            seconds=3,
            microseconds=400000,
        )

        self.assertAlmostEqual(
            date.timedelta_to_days(duration),
            2 + 3.4 / 86400,
        )

    def test_timedelta_conversion_supports_negative_values(self):
        """Negative timedeltas should convert correctly."""
        duration = date.dt.timedelta(
            days=-2,
            seconds=3600,
        )

        expected = -1.9583333333333333

        self.assertAlmostEqual(
            date.timedelta_to_days(duration),
            expected,
        )

    def test_gregorian_calendar_transition(self):
        """
        Verify the Julian/Gregorian calendar transition.

        October 4, 1582 was followed by October 15, 1582.
        """
        self.assertEqual(
            date.date_to_jd(1582, 10, 4.0),
            2299159.5,
        )

        self.assertEqual(
            date.date_to_jd(1582, 10, 15.0),
            2299160.5,
        )

        self.assertEqual(
            date.jd_to_date(2299159.5),
            (1582, 10, 4.0),
        )

        self.assertEqual(
            date.jd_to_date(2299160.5),
            (1582, 10, 15.0),
        )

    def test_gregorian_transition_is_one_day_apart_in_jd(self):
        """
        The calendar skips ten civil dates, but JD remains continuous.
        """
        jd_before = date.date_to_jd(
            1582,
            10,
            4.0,
        )

        jd_after = date.date_to_jd(
            1582,
            10,
            15.0,
        )

        self.assertEqual(
            jd_after - jd_before,
            1.0,
        )

    def test_jd_date_round_trip_multiple_epochs(self):
        """Calendar dates should survive JD round trips across epochs."""
        test_dates = (
            (1900, 1, 1.0),
            (1950, 6, 15.5),
            (1970, 1, 1.0),
            (2000, 1, 1.5),
            (2020, 2, 29.25),
            (2024, 12, 31.75),
            (2050, 7, 4.125),
        )

        for year, month, day in test_dates:
            jd = date.date_to_jd(
                year,
                month,
                day,
            )

            recovered = date.jd_to_date(jd)

            self.assertAlmostEqual(
                recovered[0],
                year,
            )

            self.assertEqual(
                recovered[1],
                month,
            )

            self.assertAlmostEqual(
                recovered[2],
                day,
                places=10,
            )

    def test_jd_to_date2_formats_known_date(self):
        """Formatted date output should match the expected representation."""
        self.assertEqual(
            date.jd_to_date2(2446113.75),
            "Feb 17 1985  06:00:00.000000",
        )

    def test_jd_to_date2_formats_midnight(self):
        """Formatted output should correctly represent midnight."""
        self.assertEqual(
            date.jd_to_date2(2451544.5),
            "Jan 1 2000  00:00:00.000000",
        )

    def test_date_spj2000_reference_epoch(self):
        """J2000 must map to zero seconds."""
        self.assertEqual(
            date.Date_SPJ2000(1.5, 1, 2000),
            0.0,
        )

    def test_date_spj2000_one_day_after_epoch(self):
        """One day after J2000 must equal 86400 seconds."""
        self.assertEqual(
            date.Date_SPJ2000(2.5, 1, 2000),
            86400.0,
        )

    def test_date_spj2000_one_day_before_epoch(self):
        """One day before J2000 must equal -86400 seconds."""
        self.assertEqual(
            date.Date_SPJ2000(31.5, 12, 1999),
            -86400.0,
        )

    def test_jd_spj2000_positive_and_negative_offsets(self):
        """J2000 seconds conversion must work in both directions."""
        self.assertEqual(
            date.JD_SPJ2000D(2451546.0),
            86400.0,
        )

        self.assertEqual(
            date.JD_SPJ2000D(2451544.0),
            -86400.0,
        )

    def test_datetime_addition(self):
        """Datetime addition must preserve calendar behavior."""
        start = date.datetime(
            2024,
            2,
            28,
            6,
            0,
            0,
        )

        result = start + date.dt.timedelta(days=1)

        self.assertEqual(
            result,
            date.datetime(
                2024,
                2,
                29,
                6,
                0,
                0,
            ),
        )

    def test_datetime_subtraction(self):
        """Datetime subtraction must produce the correct timedelta."""
        start = date.datetime(
            2024,
            2,
            29,
            6,
            0,
            0,
        )

        end = date.datetime(
            2024,
            3,
            1,
            6,
            0,
            0,
        )

        self.assertEqual(
            end - start,
            date.dt.timedelta(days=1),
        )

    def test_datetime_addition_across_year_boundary(self):
        """Datetime arithmetic must handle New Year's rollover."""
        start = date.datetime(
            2023,
            12,
            31,
            23,
            0,
            0,
        )

        result = start + date.dt.timedelta(hours=2)

        self.assertEqual(
            result,
            date.datetime(
                2024,
                1,
                1,
                1,
                0,
                0,
            ),
        )

    def test_datetime_subtraction_across_leap_day(self):
        """Datetime subtraction must handle leap years."""
        start = date.datetime(
            2024,
            2,
            28,
        )

        end = date.datetime(
            2024,
            3,
            1,
        )

        self.assertEqual(
            end - start,
            date.dt.timedelta(days=2),
        )

    def test_datetime_timedelta_round_trip(self):
        """Adding and subtracting the same duration should recover the date."""
        original = date.datetime(
            2024,
            5,
            17,
            14,
            32,
            11,
            123456,
        )

        duration = date.dt.timedelta(
            days=10,
            seconds=1234,
            microseconds=567,
        )

        recovered = (
            original
            + duration
            - duration
        )

        self.assertLessEqual(
            abs(recovered - original),
            date.dt.timedelta(microseconds=10),
        )

    def test_datetime_invalid_addition_type(self):
        """Adding a non-timedelta should raise TypeError."""
        original = date.datetime(
            2024,
            1,
            1,
        )

        with self.assertRaises(TypeError):
            original + 1

    def test_datetime_invalid_subtraction_type(self):
        """Subtracting an unsupported type should raise TypeError."""
        original = date.datetime(
            2024,
            1,
            1,
        )

        with self.assertRaises(TypeError):
            original - 1

    def test_datetime_to_jd_and_mjd(self):
        """Datetime JD and MJD methods should agree with standalone functions."""
        original = date.datetime(
            2024,
            1,
            1,
            12,
            0,
            0,
        )

        self.assertEqual(
            original.to_jd(),
            date.datetime_to_jd(original),
        )

        self.assertEqual(
            original.to_mjd(),
            date.jd_to_mjd(
                date.datetime_to_jd(original)
            ),
        )

if __name__ == "__main__":
    unittest.main()