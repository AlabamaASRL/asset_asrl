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
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertEqual(frame.mu, 1)

        self.assertAlmostEqual(
            frame.tstar,
            np.sqrt(frame.lstar**3 / frame.P1mu),
        )

        self.assertAlmostEqual(
            frame.vstar,
            np.sqrt(frame.P1mu / frame.lstar),
        )

        self.assertAlmostEqual(
            frame.astar,
            frame.vstar / frame.tstar,
        )

        self.assertAlmostEqual(
            frame.astar,
            frame.P1mu / frame.lstar**2,
        )

        self.assertAlmostEqual(
            frame.mustar,
            constants.MuEarth,
        )

    def test_characteristic_units_are_self_consistent(self):
        """
        Verify characteristic velocity and acceleration follow from the
        length and time scales.

        The expected relationships are:

            v* = l* / t*
            a* = l* / t*^2
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.lstar / frame.tstar,
            frame.vstar,
        )

        self.assertAlmostEqual(
            frame.lstar / frame.tstar**2,
            frame.astar,
        )

    def test_nondimensional_gravitational_parameter_is_unity(self):
        """
        Verify characteristic scaling normalizes the gravitational parameter
        to one.

        The nondimensional gravitational parameter is:

            mu_nd = mu * t*^2 / l*^3
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        nondimensional_mu = (
            frame.P1mu
            * frame.tstar**2
            / frame.lstar**3
        )

        self.assertAlmostEqual(
            nondimensional_mu,
            1.0,
        )

    def test_circular_orbit_has_unit_nondimensional_speed_and_period(self):
        """
        Verify a circular orbit at the characteristic radius has unit
        nondimensional speed and a nondimensional period of 2*pi.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        circular_speed = np.sqrt(
            constants.MuEarth / constants.RadiusEarth
        )

        circular_period = (
            2.0
            * np.pi
            * np.sqrt(
                constants.RadiusEarth**3
                / constants.MuEarth
            )
        )

        self.assertAlmostEqual(
            circular_speed / frame.vstar,
            1.0,
        )

        self.assertAlmostEqual(
            circular_period / frame.tstar,
            2.0 * np.pi,
        )

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
        circular_speed = np.sqrt(
            constants.MuEarth / constants.RadiusEarth
        )

        escape_speed = np.sqrt(
            2.0
            * constants.MuEarth
            / constants.RadiusEarth
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
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        radius = constants.RadiusEarth

        circular_speed = np.sqrt(
            constants.MuEarth / radius
        )

        specific_energy = (
            0.5 * circular_speed**2
            - constants.MuEarth / radius
        )

        expected_energy = (
            -constants.MuEarth
            / (2.0 * radius)
        )

        self.assertAlmostEqual(
            specific_energy,
            expected_energy,
        )

        nondimensional_energy = (
            specific_energy / frame.vstar**2
        )

        self.assertAlmostEqual(
            nondimensional_energy,
            -0.5,
        )

    def test_escape_orbit_has_zero_specific_energy(self):
        """
        Verify escape velocity produces approximately zero specific energy.
        """
        radius = constants.RadiusEarth

        escape_speed = np.sqrt(
            2.0
            * constants.MuEarth
            / radius
        )

        specific_energy = (
            0.5 * escape_speed**2
            - constants.MuEarth / radius
        )

        self.assertAlmostEqual(
            specific_energy,
            0.0,
            delta=1.0e-8,
        )

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
        frame1 = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        frame2 = TwoBodyFrame(
            constants.MuEarth,
            2.0 * constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame2.tstar / frame1.tstar,
            2.0**1.5,
        )

        self.assertAlmostEqual(
            frame2.vstar / frame1.vstar,
            1.0 / np.sqrt(2.0),
        )

        self.assertAlmostEqual(
            frame2.astar / frame1.astar,
            0.25,
        )

    def test_one_au_solar_orbit_has_about_one_year_period(self):
        """
        Verify a one-AU solar orbit has approximately a one-year period.

        The expected orbital period is between 365 and 366 days.
        """
        frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )

        orbital_period_days = (
            2.0
            * np.pi
            * frame.tstar
            / constants.day
        )

        self.assertGreater(
            orbital_period_days,
            365.0,
        )

        self.assertLess(
            orbital_period_days,
            366.0,
        )

    def test_solar_characteristic_velocity_matches_earth_orbital_speed_scale(self):
        """
        Verify the one-AU solar velocity scale is near Earth's orbital speed.

        Earth's average heliocentric orbital speed is approximately 29.8 km/s.
        """
        frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )

        earth_orbital_speed_ms = frame.vstar

        self.assertGreater(
            earth_orbital_speed_ms,
            29_000.0,
        )

        self.assertLess(
            earth_orbital_speed_ms,
            31_000.0,
        )

    # ================================================================
    # Additional physics regression tests
    # ================================================================

    def test_characteristic_length_is_preserved(self):
        """
        Verify the characteristic length is the supplied reference length.

        This is important because all other characteristic scales depend
        directly on l*.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.lstar,
            constants.RadiusEarth,
        )

    def test_characteristic_gravitational_parameter_is_preserved(self):
        """
        Verify the dimensional gravitational parameter is retained.

        The frame should retain the supplied physical mu as mustar.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.mustar,
            constants.MuEarth,
        )

    def test_kepler_third_law(self):
        """
        Verify Kepler's third law using the characteristic scales.

        For a circular orbit:

            T^2 = 4*pi^2*r^3/mu

        At the characteristic radius, this should reduce to:

            T / t* = 2*pi
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        period = (
            2.0
            * np.pi
            * np.sqrt(
                frame.lstar**3 / frame.P1mu
            )
        )

        self.assertAlmostEqual(
            period / frame.tstar,
            2.0 * np.pi,
        )

    def test_vis_viva_equation_at_multiple_radii(self):
        """
        Verify the characteristic velocity is consistent with the
        vis-viva equation at several radii.

        The vis-viva equation is:

            v^2 = mu * (2/r - 1/a)

        For a circular orbit, a = r, giving:

            v^2 = mu/r
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        radii = np.array([
            constants.RadiusEarth,
            2.0 * constants.RadiusEarth,
            3.0 * constants.RadiusEarth,
            5.0 * constants.RadiusEarth,
        ])

        for radius in radii:
            circular_speed = np.sqrt(
                constants.MuEarth / radius
            )

            semi_major_axis = radius

            vis_viva_speed = np.sqrt(
                constants.MuEarth
                * (
                    2.0 / radius
                    - 1.0 / semi_major_axis
                )
            )

            self.assertAlmostEqual(
                circular_speed,
                vis_viva_speed,
            )

        # Also verify that the characteristic radius has unit
        # nondimensional circular speed.
        characteristic_speed = np.sqrt(
            frame.P1mu / frame.lstar
        )

        self.assertAlmostEqual(
            characteristic_speed / frame.vstar,
            1.0,
        )

    def test_nondimensional_circular_orbit_energy_is_independent_of_mu(self):
        """
        Verify that characteristic scaling produces the same nondimensional
        circular-orbit energy for different gravitational systems.

        For a circular orbit:

            epsilon = -mu/(2r)

        Dividing by v*^2 = mu/r gives:

            epsilon_nd = -1/2
        """
        earth_frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        solar_frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )

        for frame in (earth_frame, solar_frame):
            specific_energy = (
                -frame.P1mu
                / (2.0 * frame.lstar)
            )

            nondimensional_energy = (
                specific_energy
                / frame.vstar**2
            )

            self.assertAlmostEqual(
                nondimensional_energy,
                -0.5,
            )

    def test_velocity_scale_changes_with_gravitational_parameter(self):
        """
        Verify the expected square-root dependence of velocity on mu.

        Holding characteristic length fixed:

            v* = sqrt(mu/l*)

        Therefore, multiplying mu by four should multiply v* by two.
        """
        radius = constants.RadiusEarth

        frame1 = TwoBodyFrame(
            constants.MuEarth,
            radius,
        )

        frame2 = TwoBodyFrame(
            4.0 * constants.MuEarth,
            radius,
        )

        self.assertAlmostEqual(
            frame2.vstar / frame1.vstar,
            2.0,
        )

    def test_time_scale_changes_with_gravitational_parameter(self):
        """
        Verify the expected inverse square-root dependence of time on mu.

        Holding characteristic length fixed:

            t* = sqrt(l*^3/mu)

        Therefore, multiplying mu by four should divide t* by two.
        """
        radius = constants.RadiusEarth

        frame1 = TwoBodyFrame(
            constants.MuEarth,
            radius,
        )

        frame2 = TwoBodyFrame(
            4.0 * constants.MuEarth,
            radius,
        )

        self.assertAlmostEqual(
            frame2.tstar / frame1.tstar,
            0.5,
        )

    def test_acceleration_scale_changes_with_gravitational_parameter(self):
        """
        Verify acceleration has the expected linear dependence on mu.

        Holding characteristic length fixed:

            a* = mu/l*^2

        Therefore, multiplying mu by four should multiply a* by four.
        """
        radius = constants.RadiusEarth

        frame1 = TwoBodyFrame(
            constants.MuEarth,
            radius,
        )

        frame2 = TwoBodyFrame(
            4.0 * constants.MuEarth,
            radius,
        )

        self.assertAlmostEqual(
            frame2.astar / frame1.astar,
            4.0,
        )

    def test_scaling_invariant_under_simultaneous_length_and_mu_change(self):
        """
        Verify that characteristic scaling correctly adapts when both
        gravitational parameter and length scale are changed.

        Starting with:

            l2 = 4*l1
            mu2 = 4*mu1

        gives:

            t2/t1 = 4
            v2/v1 = 1
            a2/a1 = 1/4
        """
        frame1 = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        frame2 = TwoBodyFrame(
            4.0 * constants.MuEarth,
            4.0 * constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame2.tstar / frame1.tstar,
            4.0,
        )

        self.assertAlmostEqual(
            frame2.vstar / frame1.vstar,
            1.0,
        )

        self.assertAlmostEqual(
            frame2.astar / frame1.astar,
            0.25,
        )

    def test_earth_and_sun_frames_both_normalize_mu(self):
        """
        Verify that both Earth-centered and Sun-centered frames produce
        a nondimensional gravitational parameter of one.
        """
        earth_frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        sun_frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )

        for frame in (earth_frame, sun_frame):
            nondimensional_mu = (
                frame.P1mu
                * frame.tstar**2
                / frame.lstar**3
            )

            self.assertAlmostEqual(
                nondimensional_mu,
                1.0,
            )

    def test_circular_acceleration_matches_gravity(self):
        """
        Verify that the characteristic acceleration agrees with the
        gravitational acceleration at the characteristic radius.

        At r = l*:

            g = mu/l*^2

        which must equal a*.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        gravitational_acceleration = (
            constants.MuEarth
            / constants.RadiusEarth**2
        )

        self.assertAlmostEqual(
            frame.astar,
            gravitational_acceleration,
        )

    def test_velocity_acceleration_time_identity(self):
        """
        Verify the dimensional identity:

            a* * t* = v*

        This provides an independent consistency check on all three
        characteristic scales.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.astar * frame.tstar,
            frame.vstar,
        )

    def test_length_acceleration_velocity_identity(self):
        """
        Verify the dimensional identity:

            v*^2 = a* * l*
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.vstar**2,
            frame.astar * frame.lstar,
        )


if __name__ == "__main__":
    unittest.main()