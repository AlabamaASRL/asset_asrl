# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

import numpy as np

from asset_asrl.Astro import Constants as constants
from asset_asrl.Astro.Extensions.TwoBodyFrame import TwoBodyFrame


class TwoBodyFramePhysicsTests(unittest.TestCase):
    """
    Physics regression tests for two-body characteristic scaling.

    These tests verify that ``TwoBodyFrame`` satisfies the expected
    Keplerian relationships used to nondimensionalize a two-body
    gravitational system.

    The characteristic quantities satisfy:

        t* = sqrt(l*^3 / mu)
        v* = sqrt(mu / l*)
        a* = v* / t* = mu / l*^2

    where:

        l* = characteristic length
        t* = characteristic time
        v* = characteristic velocity
        a* = characteristic acceleration
        mu = gravitational parameter

    Reference
    ---------
    [1] Bate, Mueller, and White, Fundamentals of Astrodynamics.
    """

    def test_earth_surface_scales_match_keplerian_relations(self):
        """
        Verify Earth-based characteristic scales satisfy Keplerian identities.
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)

        self.assertEqual(frame.mu, 1)
        self.assertAlmostEqual(frame.tstar, np.sqrt(frame.lstar**3 / frame.P1mu))
        self.assertAlmostEqual(frame.vstar, np.sqrt(frame.P1mu / frame.lstar))
        self.assertAlmostEqual(frame.astar, frame.vstar / frame.tstar)
        self.assertAlmostEqual(frame.astar, frame.P1mu / frame.lstar**2)
        self.assertAlmostEqual(frame.mustar, constants.MuEarth)

    def test_characteristic_units_are_self_consistent(self):
        """
        Verify characteristic velocity and acceleration follow from the
        length and time scales.

        The expected relationships are:

            v* = l* / t*
            a* = l* / t*^2
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)

        self.assertAlmostEqual(frame.lstar / frame.tstar, frame.vstar)
        self.assertAlmostEqual(frame.lstar / frame.tstar**2, frame.astar)

    def test_nondimensional_gravitational_parameter_is_unity(self):
        """
        Verify characteristic scaling normalizes the gravitational parameter
        to one.

        The nondimensional gravitational parameter is:

            mu_nd = mu * t*^2 / l*^3
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)

        nondimensional_mu = frame.P1mu * frame.tstar**2 / frame.lstar**3

        self.assertAlmostEqual(nondimensional_mu, 1.0)

    def test_circular_orbit_has_unit_nondimensional_speed_and_period(self):
        """
        Verify a circular orbit at the characteristic radius has unit
        nondimensional speed and a nondimensional period of 2*pi.
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)

        circular_speed = np.sqrt(constants.MuEarth / constants.RadiusEarth)
        circular_period = 2.0 * np.pi * np.sqrt(
            constants.RadiusEarth**3 / constants.MuEarth
        )

        self.assertAlmostEqual(circular_speed / frame.vstar, 1.0)
        self.assertAlmostEqual(circular_period / frame.tstar, 2.0 * np.pi)

    def test_escape_speed_is_sqrt_two_times_circular_speed(self):
        """
        Verify the standard two-body escape-speed relationship.

        For a circular orbit:

            v_circular = sqrt(mu / r)

        The local escape speed is:

            v_escape = sqrt(2*mu / r)

        Therefore:

            v_escape / v_circular = sqrt(2)
        """
        circular_speed = np.sqrt(constants.MuEarth / constants.RadiusEarth)
        escape_speed = np.sqrt(
            2.0 * constants.MuEarth / constants.RadiusEarth
        )

        self.assertAlmostEqual(
            escape_speed / circular_speed,
            np.sqrt(2.0),
        )

    def test_circular_orbit_specific_energy_matches_keplerian_value(self):
        """
        Verify circular-orbit specific energy matches the Keplerian result.

        The specific mechanical energy is:

            epsilon = v^2 / 2 - mu / r

        For a circular orbit:

            epsilon = -mu / (2*r)
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)

        radius = constants.RadiusEarth
        circular_speed = np.sqrt(constants.MuEarth / radius)

        specific_energy = 0.5 * circular_speed**2 - constants.MuEarth / radius
        expected_energy = -constants.MuEarth / (2.0 * radius)

        self.assertAlmostEqual(specific_energy, expected_energy)

        nondimensional_energy = specific_energy / frame.vstar**2

        self.assertAlmostEqual(nondimensional_energy, -0.5)

    def test_escape_orbit_has_zero_specific_energy(self):
        """
        Verify escape velocity produces approximately zero specific energy.
        """
        radius = constants.RadiusEarth
        escape_speed = np.sqrt(2.0 * constants.MuEarth / radius)

        specific_energy = 0.5 * escape_speed**2 - constants.MuEarth / radius

        self.assertAlmostEqual(specific_energy, 0.0, delta=1.0e-8)

    def test_characteristic_scales_change_consistently_with_length_scale(self):
        """
        Verify characteristic scales follow the expected power laws.

        Holding the gravitational parameter fixed:

            t* proportional to l*^(3/2)
            v* proportional to l*^(-1/2)
            a* proportional to l*^(-2)

        Doubling the characteristic length provides a regression test for
        all three scaling relationships.
        """
        frame1 = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth)
        frame2 = TwoBodyFrame(constants.MuEarth, 2.0 * constants.RadiusEarth)

        self.assertAlmostEqual(frame2.tstar / frame1.tstar, 2.0**1.5)
        self.assertAlmostEqual(
            frame2.vstar / frame1.vstar,
            1.0 / np.sqrt(2.0),
        )
        self.assertAlmostEqual(frame2.astar / frame1.astar, 0.25)

    def test_one_au_solar_orbit_has_about_one_year_period(self):
        """
        Verify a one-AU solar orbit has approximately a one-year period.

        The expected orbital period is between 365 and 366 days.
        """
        frame = TwoBodyFrame(constants.MuSun, constants.AU)

        orbital_period_days = 2.0 * np.pi * frame.tstar / constants.day

        self.assertGreater(orbital_period_days, 365.0)
        self.assertLess(orbital_period_days, 366.0)

    def test_solar_characteristic_velocity_matches_earth_orbital_speed_scale(
        self,
    ):
        """
        Verify the one-AU solar velocity scale is near Earth's orbital speed.

        Earth's average heliocentric orbital speed is approximately 29.8 km/s.
        """
        frame = TwoBodyFrame(constants.MuSun, constants.AU)

        earth_orbital_speed_ms = frame.vstar

        self.assertGreater(earth_orbital_speed_ms, 29_000.0)
        self.assertLess(earth_orbital_speed_ms, 31_000.0)


if __name__ == "__main__":
    unittest.main()