# -*- coding: utf-8 -*-

"""
Physics-facing regression tests for astronomical dates and Julian dates.

These tests validate the numerical and calendar behavior of the local
Astro/Date.py implementation, including:

1. Julian Date and Modified Julian Date conversions.
2. J2000 epoch conversions.
3. Fractional-day and time-of-day conversions.
4. Datetime <-> Julian Date conversions.
5. Gregorian calendar transition behavior.
6. Julian Date formatted output.
7. Datetime and timedelta arithmetic.
8. Round-trip preservation of microsecond precision.

The local Date.py source file is loaded directly so that these tests exercise
the source tree under development rather than a pip-installed ASSET package.
"""

import importlib.util
import pathlib
import sys
import unittest


# ============================================================================
# Local module loading
# ============================================================================

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATE_MODULE = PACKAGE_ROOT / "Astro" / "Date.py"


def load_module(name, path):
    """Load a local Python module directly from its source file."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


date = load_module("local_asset_date", DATE_MODULE)


# ============================================================================
# Julian Date reference and conversion tests
# ============================================================================

class JulianDateReferenceTests(unittest.TestCase):
    """
    Validate fundamental Julian Date and Modified Julian Date relationships.

    These tests establish reference values and verify that the primary
    calendar/JD conversion functions are mutually consistent.
    """

    def test_j2000_reference_epoch(self):
        """January 1.5, 2000 corresponds exactly to J2000."""
        jd = date.date_to_jd(2000, 1, 1.5)
        self.assertAlmostEqual(jd, 2451545.0, places=10)

    def test_j2000_seconds_reference(self):
        """J2000 Julian Date corresponds to zero seconds from J2000."""
        seconds = date.JD_SPJ2000D(2451545.0)
        self.assertAlmostEqual(seconds, 0.0, places=10)

    def test_duffet_smith_known_example(self):
        """Validate the standard Duffet-Smith Julian Date example."""
        jd = date.date_to_jd(1985, 2, 17.25)
        self.assertAlmostEqual(jd, 2446113.75, places=10)

    def test_duffet_smith_inverse(self):
        """Validate the inverse conversion of the known Julian Date."""
        year, month, day = date.jd_to_date(2446113.75)
        self.assertEqual(year, 1985)
        self.assertEqual(month, 2)
        self.assertAlmostEqual(day, 17.25, places=10)

    def test_mjd_to_jd_inverse(self):
        """MJD to JD and JD to MJD conversions are inverse operations."""
        mjd_values = (0.0, 40587.0, 51544.5, 58000.25, 60000.75)

        for mjd in mjd_values:
            jd = date.mjd_to_jd(mjd)
            recovered = date.jd_to_mjd(jd)
            self.assertAlmostEqual(recovered, mjd, places=10)

    def test_jd_mjd_offset(self):
        """JD and MJD differ by the standard 2400000.5-day offset."""
        jd = 2451545.0
        mjd = date.jd_to_mjd(jd)
        self.assertAlmostEqual(mjd, 51544.5, places=10)

    def test_jd_spj2000_positive_one_day(self):
        """One day after J2000 corresponds to 86400 seconds."""
        seconds = date.JD_SPJ2000D(2451546.0)
        self.assertAlmostEqual(seconds, 86400.0, places=10)

    def test_jd_spj2000_negative_one_day(self):
        """One day before J2000 corresponds to -86400 seconds."""
        seconds = date.JD_SPJ2000D(2451544.0)
        self.assertAlmostEqual(seconds, -86400.0, places=10)

    def test_date_spj2000_midnight(self):
        """January 1, 2000 midnight is 43200 seconds before J2000."""
        seconds = date.Date_SPJ2000(1, 1, 2000)
        self.assertAlmostEqual(seconds, -43200.0, places=10)

    def test_date_spj2000_j2000_epoch(self):
        """January 1, 2000 at noon corresponds to the J2000 epoch."""
        seconds = date.Date_SPJ2000(1.5, 1, 2000)
        self.assertAlmostEqual(seconds, 0.0, places=10)


# ============================================================================
# Fractional day and time-of-day tests
# ============================================================================

class TimeOfDayConversionTests(unittest.TestCase):
    """
    Validate conversion between fractional days and HMSM representations.

    The fractional-day representation is the interface between calendar
    dates and time-of-day values in the Julian Date algorithms.
    """

    def test_days_to_hmsm_fractional_day(self):
        """0.1 day corresponds to 2:24:00.000000."""
        result = date.days_to_hmsm(0.1)
        self.assertEqual(result, (2, 24, 0, 0))

    def test_days_to_hmsm_midnight(self):
        """Zero fractional days corresponds to midnight."""
        result = date.days_to_hmsm(0.0)
        self.assertEqual(result, (0, 0, 0, 0))

    def test_days_to_hmsm_six_hours(self):
        """One quarter of a day corresponds to 06:00:00."""
        result = date.days_to_hmsm(0.25)
        self.assertEqual(result, (6, 0, 0, 0))

    def test_days_to_hmsm_end_of_day(self):
        """A value immediately below one day converts to 23:59:59.999999."""
        result = date.days_to_hmsm(1.0 - 1.0 / date.MICROSECONDS_PER_DAY)
        self.assertEqual(result, (23, 59, 59, 999999))

    def test_hmsm_to_days_midnight(self):
        """Midnight corresponds to zero fractional days."""
        result = date.hmsm_to_days()
        self.assertAlmostEqual(result, 0.0, places=15)

    def test_hmsm_to_days_six_hours(self):
        """Six hours corresponds to one quarter of a day."""
        result = date.hmsm_to_days(hour=6)
        self.assertAlmostEqual(result, 0.25, places=15)

    def test_hmsm_to_days_end_of_day(self):
        """23:59:59.999999 is one microsecond before one day."""
        result = date.hmsm_to_days(hour=23, mins=59, sec=59, micro=999999)
        expected = 1.0 - 1.0 / date.MICROSECONDS_PER_DAY
        self.assertAlmostEqual(result, expected, places=15)

    def test_hmsm_fractional_day_inverse(self):
        """HMSM to fractional day and back preserves the time representation."""
        values = ((0, 0, 0, 0), (1, 30, 15, 250000), (6, 0, 0, 0), (12, 34, 56, 123456), (23, 59, 59, 999999))

        for hour, minute, second, microsecond in values:
            days = date.hmsm_to_days(hour, minute, second, microsecond)
            recovered = date.days_to_hmsm(days)
            self.assertEqual(recovered, (hour, minute, second, microsecond))


# ============================================================================
# Datetime and Julian Date conversion tests
# ============================================================================

class DatetimeJulianConversionTests(unittest.TestCase):
    """
    Validate conversions between datetime objects and Julian Dates.

    These tests emphasize round-trip accuracy, including leap days,
    year boundaries, and microsecond-level precision.
    """

    def test_datetime_to_jd_known_date(self):
        """A known datetime converts to the expected Julian Date."""
        value = date.datetime(1985, 2, 17, 6, 0, 0)
        jd = date.datetime_to_jd(value)
        self.assertAlmostEqual(jd, 2446113.75, places=10)

    def test_datetime_to_jd_midnight(self):
        """January 1, 2000 midnight has the expected Julian Date."""
        value = date.datetime(2000, 1, 1, 0, 0, 0)
        jd = date.datetime_to_jd(value)
        self.assertAlmostEqual(jd, 2451544.5, places=10)

    def test_jd_to_datetime_j2000(self):
        """J2000 converts to January 1, 2000 at noon."""
        result = date.jd_to_datetime(2451545.0)
        expected = date.datetime(2000, 1, 1, 12, 0, 0)
        self.assertEqual(result, expected)

    def test_datetime_jd_round_trip(self):
        """Datetime -> JD -> datetime preserves representative timestamps."""
        values = (date.datetime(2000, 1, 1, 12, 0, 0), date.datetime(1985, 2, 17, 6, 0, 0), date.datetime(2024, 2, 29, 12, 30, 45), date.datetime(2023, 12, 31, 23, 59, 59))

        for value in values:
            jd = date.datetime_to_jd(value)
            recovered = date.jd_to_datetime(jd)
            difference = abs((recovered - value).total_seconds())
            self.assertLessEqual(difference, 1e-4)

    def test_datetime_jd_round_trip_microseconds(self):
        """Datetime round trips preserve microsecond-level precision."""
        values = (date.datetime(2024, 2, 29, 12, 30, 45, 123456), date.datetime(2023, 12, 31, 23, 59, 59, 999999), date.datetime(2000, 1, 1, 12, 0, 0, 1))

        for value in values:
            jd = date.datetime_to_jd(value)
            recovered = date.jd_to_datetime(jd)
            difference = abs((recovered - value).total_seconds())
            self.assertLessEqual(difference, 1e-4)

    def test_leap_day_round_trip(self):
        """Leap-day timestamps survive a JD round trip."""
        value = date.datetime(2024, 2, 29, 15, 30, 45, 500000)
        jd = date.datetime_to_jd(value)
        recovered = date.jd_to_datetime(jd)
        difference = abs((recovered - value).total_seconds())
        self.assertLessEqual(difference, 1e-4)

    def test_year_boundary_round_trip(self):
        """New Year's Eve timestamps survive a JD round trip."""
        value = date.datetime(2023, 12, 31, 23, 59, 59, 500000)
        jd = date.datetime_to_jd(value)
        recovered = date.jd_to_datetime(jd)
        difference = abs((recovered - value).total_seconds())
        self.assertLessEqual(difference, 1e-4)

    def test_datetime_to_jd_and_mjd_methods(self):
        """Datetime convenience methods agree with standalone functions."""
        value = date.datetime(2024, 6, 1, 12, 30, 15, 123456)
        self.assertAlmostEqual(value.to_jd(), date.datetime_to_jd(value), places=12)
        self.assertAlmostEqual(value.to_mjd(), date.jd_to_mjd(date.datetime_to_jd(value)), places=12)


# ============================================================================
# Calendar system and Gregorian transition tests
# ============================================================================

class CalendarSystemTests(unittest.TestCase):
    """
    Validate Julian/Gregorian calendar boundary behavior.

    The Julian Date algorithms must account for the historical Gregorian
    calendar transition on October 15, 1582.
    """

    def test_gregorian_transition_dates(self):
        """October 4 and October 15, 1582 are consecutive calendar dates."""
        jd_before = date.date_to_jd(1582, 10, 4)
        jd_after = date.date_to_jd(1582, 10, 15)
        self.assertAlmostEqual(jd_after - jd_before, 1.0, places=10)

    def test_gregorian_transition_inverse(self):
        """The transition Julian Dates recover the expected calendar dates."""
        year1, month1, day1 = date.jd_to_date(date.date_to_jd(1582, 10, 4))
        year2, month2, day2 = date.jd_to_date(date.date_to_jd(1582, 10, 15))
        self.assertEqual((year1, month1), (1582, 10))
        self.assertAlmostEqual(day1, 4.0, places=10)
        self.assertEqual((year2, month2), (1582, 10))
        self.assertAlmostEqual(day2, 15.0, places=10)

    def test_jd_date_round_trip_multiple_epochs(self):
        """Calendar dates round-trip through Julian Date conversion."""
        values = ((1900, 1, 1.0), (1950, 6, 15.5), (2000, 1, 1.5), (2024, 2, 29.25), (2025, 12, 31.75))

        for year, month, day in values:
            with self.subTest(year=year, month=month, day=day):
                jd = date.date_to_jd(year, month, day)
                recovered_year, recovered_month, recovered_day = date.jd_to_date(jd)
                self.assertEqual(recovered_year, year)
                self.assertEqual(recovered_month, month)
                self.assertAlmostEqual(recovered_day, day, places=9)


# ============================================================================
# Formatted Julian Date tests
# ============================================================================

class FormattedDateTests(unittest.TestCase):
    """
    Validate human-readable Julian Date formatting.

    These tests verify that jd_to_date2() produces consistently padded
    hour/minute/second/microsecond fields.
    """

    def test_jd_to_date2_formats_known_date(self):
        """Known Julian Date produces the expected formatted representation."""
        result = date.jd_to_date2(2446113.75)
        self.assertEqual(result, "Feb 17 1985  06:00:00.000000")

    def test_jd_to_date2_formats_midnight(self):
        """Midnight is formatted with zero-padded time fields."""
        result = date.jd_to_date2(2451544.5)
        self.assertEqual(result, "Jan 1 2000  00:00:00.000000")

    def test_jd_to_date2_formats_fractional_seconds(self):
        """
        Fractional seconds are represented to six decimal places.

        A Julian Date is a large floating-point value, so a few microseconds
        of numerical round-off are expected when converting through JD.
        """
        value = date.datetime(2024, 2, 29, 12, 34, 56, 123456)
        result = date.jd_to_date2(date.datetime_to_jd(value))
        self.assertRegex(result, r"^Feb 29 2024  12:34:56\.\d{6}$")
        actual_microseconds = int(result[-6:])
        self.assertLessEqual(abs(actual_microseconds - value.microsecond), 10)


# ============================================================================
# Datetime arithmetic tests
# ============================================================================

class DatetimeArithmeticTests(unittest.TestCase):
    """
    Validate arithmetic behavior of the custom datetime subclass.

    Native datetime arithmetic is used for timedelta operations while
    preserving the custom datetime subclass.
    """

    def test_datetime_addition(self):
        """Adding a timedelta produces the expected datetime."""
        value = date.datetime(2024, 1, 1, 12, 0, 0)
        result = value + date.dt.timedelta(hours=6)
        self.assertEqual(result, date.datetime(2024, 1, 1, 18, 0, 0))

    def test_datetime_subtraction(self):
        """Subtracting a timedelta produces the expected datetime."""
        value = date.datetime(2024, 1, 2, 12, 0, 0)
        result = value - date.dt.timedelta(hours=6)
        self.assertEqual(result, date.datetime(2024, 1, 2, 6, 0, 0))

    def test_datetime_addition_across_year_boundary(self):
        """Datetime arithmetic correctly handles New Year's rollover."""
        value = date.datetime(2023, 12, 31, 23, 0, 0)
        result = value + date.dt.timedelta(hours=2)
        expected = date.datetime(2024, 1, 1, 1, 0, 0)
        difference = abs((result - expected).total_seconds())
        self.assertLessEqual(difference, 1e-4)

    def test_datetime_addition_across_leap_day(self):
        """Datetime arithmetic correctly handles leap-day transitions."""
        value = date.datetime(2024, 2, 28, 23, 0, 0)
        result = value + date.dt.timedelta(hours=2)
        self.assertEqual(result, date.datetime(2024, 2, 29, 1, 0, 0))

    def test_datetime_timedelta_round_trip(self):
        """Adding and subtracting the same timedelta recovers the original."""
        value = date.datetime(2024, 7, 15, 10, 30, 45, 123456)
        delta = date.dt.timedelta(days=3, hours=7, minutes=12, seconds=4, microseconds=789)
        result = value + delta - delta
        self.assertEqual(result, value)

    def test_datetime_addition_rejects_invalid_type(self):
        """Invalid addition operands are rejected by datetime arithmetic."""
        value = date.datetime(2024, 1, 1)
        with self.assertRaises(TypeError):
            value + 1

    def test_datetime_subtraction_rejects_invalid_type(self):
        """Invalid subtraction operands are rejected by datetime arithmetic."""
        value = date.datetime(2024, 1, 1)
        with self.assertRaises(TypeError):
            value - 1


# ============================================================================
# Timedelta conversion tests
# ============================================================================

class TimedeltaConversionTests(unittest.TestCase):
    """Validate conversion from datetime.timedelta to fractional days."""

    def test_timedelta_conversion_integer_days(self):
        """A whole number of timedelta days converts exactly."""
        duration = date.dt.timedelta(days=4)
        result = date.timedelta_to_days(duration)
        self.assertAlmostEqual(result, 4.0, places=15)

    def test_timedelta_conversion_fractional_day(self):
        """Hours are correctly represented as a fractional day."""
        duration = date.dt.timedelta(days=2, hours=12)
        result = date.timedelta_to_days(duration)
        self.assertAlmostEqual(result, 2.5, places=15)

    def test_timedelta_conversion_preserves_seconds(self):
        """Fractional seconds are preserved in the day conversion."""
        duration = date.dt.timedelta(days=2, seconds=3.4)
        result = date.timedelta_to_days(duration)
        expected = 2.0 + 3.4 / date.SECONDS_PER_DAY
        self.assertAlmostEqual(result, expected, places=15)

    def test_timedelta_conversion_preserves_microseconds(self):
        """Microseconds contribute correctly to the fractional day."""
        duration = date.dt.timedelta(days=2, seconds=3, microseconds=400000)
        result = date.timedelta_to_days(duration)
        expected = 2.0 + 3.4 / date.SECONDS_PER_DAY
        self.assertAlmostEqual(result, expected, places=15)

    def test_timedelta_conversion_rejects_invalid_type(self):
        """Non-timedelta inputs are rejected."""
        with self.assertRaises((TypeError, AttributeError)):
            date.timedelta_to_days(2.0)


# ============================================================================
# Test runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)