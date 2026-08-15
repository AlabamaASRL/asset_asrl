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
    # Load these self-contained modules directly to avoid requiring the native extension.
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


date = load_module("asset_asrl_astro_date", DATE_MODULE)
constants = load_module("asset_asrl_astro_constants", CONSTANTS_MODULE)


class JulianDateTests(unittest.TestCase):
    def test_reference_epoch_j2000(self):
        # J2000 is the standard epoch used by the orbital routines.
        self.assertEqual(date.date_to_jd(2000, 1, 1.5), 2451545.0)
        self.assertEqual(date.JD_SPJ2000D(2451545.0), 0.0)

    def test_known_astronomical_example(self):
        self.assertEqual(date.date_to_jd(1985, 2, 17.25), 2446113.75)
        self.assertEqual(date.jd_to_date(2446113.75), (1985, 2, 17.25))

    def test_julian_and_modified_julian_day_are_inverse_conversions(self):
        for jd in (0.0, 2400000.5, 2451545.0, 2460000.25):
            self.assertEqual(date.mjd_to_jd(date.jd_to_mjd(jd)), jd)

    def test_calendar_datetime_round_trip_preserves_microseconds(self):
        original = date.datetime(2024, 2, 29, 12, 34, 56, 123456)
        recovered = date.jd_to_datetime(date.datetime_to_jd(original))
        # Julian dates are represented by IEEE-754 floats; allow the few
        # microseconds of quantization error at modern epochs.
        self.assertLessEqual(abs(recovered - original), date.dt.timedelta(microseconds=10))

    def test_timedelta_conversion_preserves_microseconds(self):
        duration = date.dt.timedelta(days=2, seconds=3, microseconds=400000)
        self.assertAlmostEqual(date.timedelta_to_days(duration), 2 + 3.4 / 86400)

    def test_hmsm_and_fractional_day_conversions_are_inverse(self):
        fraction = date.hmsm_to_days(hour=23, mins=59, sec=59, micro=500000)
        self.assertAlmostEqual(fraction, (23 * 3600 + 59 * 60 + 59.5) / 86400)
        self.assertEqual(date.days_to_hmsm(fraction), (23, 59, 59, 500000))

    def test_gregorian_calendar_transition_uses_known_julian_days(self):
        # The Julian/Gregorian switchover is a common source of off-by-ten-day errors.
        self.assertEqual(date.date_to_jd(1582, 10, 4.0), 2299159.5)
        self.assertEqual(date.date_to_jd(1582, 10, 15.0), 2299160.5)
        self.assertEqual(date.jd_to_date(2299159.5), (1582, 10, 4.0))
        self.assertEqual(date.jd_to_date(2299160.5), (1582, 10, 15.0))

    def test_date_spj2000_is_seconds_from_j2000_epoch(self):
        # One civil day must map to exactly 86,400 seconds from the reference epoch.
        self.assertEqual(date.Date_SPJ2000(1.5, 1, 2000), 0.0)
        self.assertEqual(date.Date_SPJ2000(2.5, 1, 2000), 86400.0)

    def test_datetime_arithmetic_preserves_a_one_day_interval(self):
        start = date.datetime(2024, 2, 28, 6, 0, 0)
        result = start + date.dt.timedelta(days=1)
        self.assertEqual(result, date.datetime(2024, 2, 29, 6, 0, 0))
        self.assertEqual(result - start, date.dt.timedelta(days=1))


class PhysicalConstantsTests(unittest.TestCase):
    def test_earth_circular_speed_at_surface_matches_mu_over_radius(self):
        # Circular speed follows directly from v = sqrt(mu / r).
        speed = (constants.MuEarth / constants.RadiusEarth) ** 0.5
        self.assertAlmostEqual(speed, 7905.36, delta=5.0)

    def test_mass_and_gravitational_parameter_are_consistent(self):
        # Constants expose both mass and mu, so G * mass must reconstruct mu.
        for body in ("Earth", "Moon", "Mars", "Sun"):
            mu = getattr(constants, f"Mu{body}")
            mass = getattr(constants, f"{body}Mass")
            self.assertAlmostEqual(constants.Gcon * mass, mu, delta=mu * 1e-14)

    def test_earth_properties_are_exposed_to_spice_consumers(self):
        earth = constants.SpiceBodyProps["EARTH"]
        self.assertEqual(earth["Mu"], constants.MuEarth)
        self.assertEqual(earth["Radius"], constants.RadiusEarth)
        self.assertEqual(earth["J2"], constants.J2Earth)

    def test_time_and_distance_units_are_physically_consistent(self):
        self.assertEqual(constants.minute, 60 * constants.sec)
        self.assertEqual(constants.hour, 60 * constants.minute)
        self.assertEqual(constants.day, 24 * constants.hour)
        self.assertEqual(constants.year, 365 * constants.day)
        self.assertEqual(constants.kilometer, 1000 * constants.meter)

    def test_planetary_properties_are_positive_and_have_expected_ordering(self):
        for body in ("SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS BARYCENTER", "JUPITER BARYCENTER"):
            properties = constants.SpiceBodyProps[body]
            self.assertGreater(properties["Mu"], 0)
            if "Radius" in properties:
                self.assertGreater(properties["Radius"], 0)

        self.assertGreater(constants.RadiusSun, constants.RadiusJupiter)
        self.assertGreater(constants.MuJupiter, constants.MuEarth)
        self.assertGreater(constants.MuEarth, constants.MuMoon)


if __name__ == "__main__":
    unittest.main()
