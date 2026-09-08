# -*- coding: utf-8 -*-

from __future__ import annotations
import unittest
from asset_asrl.Astro import Constants as constants


class PhysicalConstantsTests(unittest.TestCase):

    def test_earth_circular_speed_at_surface(self):
        """Verify v = sqrt(mu / r) for Earth."""
        speed = (constants.MuEarth / constants.RadiusEarth) ** 0.5

        self.assertAlmostEqual(speed, 7905.36, delta=5.0)

    def test_mass_and_gravitational_parameter_are_consistent(self):
        """
        Verify mu = G*M for the supported bodies.
        """
        for body in ("Earth", "Moon", "Mars", "Sun"):
            mu = getattr(constants, f"Mu{body}")
            mass = getattr(constants, f"{body}Mass")

            self.assertAlmostEqual(constants.Gcon * mass, mu, delta=mu * 1e-14)

    def test_earth_properties_are_exposed_to_spice_consumers(self):
        """SPICE Earth properties must agree with named constants."""
        earth = constants.SpiceBodyProps["EARTH"]

        self.assertEqual(earth["Mu"], constants.MuEarth)
        self.assertEqual(earth["Radius"], constants.RadiusEarth)
        self.assertEqual(earth["J2"], constants.J2Earth)

    def test_time_units_are_consistent(self):
        """Verify the hierarchy of time units."""
        self.assertEqual(constants.minute, 60 * constants.sec)
        self.assertEqual(constants.hour, 60 * constants.minute)
        self.assertEqual(constants.day, 24 * constants.hour)
        self.assertEqual(constants.year, 365 * constants.day)

    def test_distance_units_are_consistent(self):
        """Verify SI distance conversions."""
        self.assertEqual(constants.kilometer, 1000 * constants.meter)

    def test_planetary_properties_are_positive(self):
        """Physical planetary parameters should be positive."""
        for body in (
            "SUN",
            "MERCURY",
            "VENUS",
            "EARTH",
            "MOON",
            "MARS BARYCENTER",
            "JUPITER BARYCENTER",
        ):
            properties = constants.SpiceBodyProps[body]

            self.assertGreater(properties["Mu"], 0)

            if "Radius" in properties:
                self.assertGreater(properties["Radius"], 0)

    def test_planetary_gravitational_parameter_ordering(self):
        """Check expected relative magnitudes of major bodies."""
        self.assertGreater(constants.MuJupiter, constants.MuEarth)
        self.assertGreater(constants.MuEarth, constants.MuMoon)

    def test_planetary_radius_ordering(self):
        """Check expected relative planetary radii."""
        self.assertGreater(constants.RadiusSun, constants.RadiusJupiter)
        self.assertGreater(constants.RadiusEarth, constants.RadiusMoon)

    def test_earth_escape_velocity_is_physically_reasonable(self):
        """
        Verify Earth's surface escape velocity.

        v_escape = sqrt(2 * mu / r)
        """
        escape_velocity = (2.0 * constants.MuEarth / constants.RadiusEarth) ** 0.5

        self.assertAlmostEqual(escape_velocity, 11186.0, delta=20.0)

    def test_earth_surface_gravity_is_physically_reasonable(self):
        """
        Verify Earth's surface gravitational acceleration.

        g = mu / r^2
        """
        gravity = constants.MuEarth / constants.RadiusEarth**2

        self.assertAlmostEqual(gravity, 9.798, delta=0.05)

    def test_earth_circular_speed_is_less_than_escape_speed(self):
        """Escape velocity should equal sqrt(2) times circular velocity."""
        circular_speed = (constants.MuEarth / constants.RadiusEarth) ** 0.5
        escape_speed = (2.0 * constants.MuEarth / constants.RadiusEarth) ** 0.5

        self.assertAlmostEqual(escape_speed / circular_speed, 2.0**0.5, places=12)


if __name__ == "__main__":
    unittest.main()