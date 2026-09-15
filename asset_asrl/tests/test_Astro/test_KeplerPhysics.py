# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import unittest

import asset as ast
import numpy as np


# ======================================================================
# Shared Kepler propagator utilities
# ======================================================================

class KeplerPropagatorTestBase(unittest.TestCase):
    """
    Shared setup and physics utilities for Kepler propagator tests.

    The tests use a nondimensional gravitational parameter:

        mu = 1

    This produces simple analytical reference solutions for circular
    and escape trajectories.
    """

    mu = 1.0
    propagator = None

    @classmethod
    def setUpClass(cls):
        """
        Construct the ASSET Kepler propagator used by the test classes.

        For a unit circular orbit:

            r = 1
            mu = 1
            v = sqrt(mu / r) = 1

        The corresponding orbital period is:

            T = 2*pi
        """
        cls.propagator = ast.Astro.Kepler.KeplerPropagator(cls.mu).vf()

    @staticmethod
    def specific_energy(state, mu):
        """
        Return the specific mechanical energy.

        E = 1/2 * |v|^2 - mu / |r|
        """
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)

        return 0.5 * np.dot(velocity, velocity) - mu / np.linalg.norm(position)

    @staticmethod
    def specific_angular_momentum(state):
        """
        Return the specific angular momentum vector.

        h = r x v
        """
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)

        return np.cross(position, velocity)

    @staticmethod
    def orbital_radius(state):
        """
        Return the instantaneous orbital radius.
        """
        return np.linalg.norm(np.asarray(state[:3], dtype=float))

    @staticmethod
    def orbital_speed(state):
        """
        Return the instantaneous orbital speed.
        """
        return np.linalg.norm(np.asarray(state[3:6], dtype=float))

    @staticmethod
    def eccentricity_vector(state, mu):
        """
        Return the eccentricity vector.

        e = (v x h)/mu - r/r
        """
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)

        angular_momentum = np.cross(position, velocity)

        return np.cross(velocity, angular_momentum) / mu - position / np.linalg.norm(position)

    @staticmethod
    def orbital_eccentricity(state, mu):
        """
        Return the scalar orbital eccentricity.
        """
        return np.linalg.norm(KeplerPropagatorTestBase.eccentricity_vector(state, mu))

    def propagate(self, state, duration):
        """
        Propagate a Cartesian state using the ASSET Kepler propagator.

        ASSET expects:

            [x, y, z, vx, vy, vz, dt]
        """
        initial_condition = np.concatenate((np.asarray(state, dtype=float), [duration]))

        return np.asarray(self.propagator.compute(initial_condition), dtype=float)


# ======================================================================
# 1. Analytical circular-orbit validation
# ======================================================================

class KeplerCircularOrbitTests(KeplerPropagatorTestBase):
    """
    Validate the Kepler propagator against analytical circular orbits.

    These tests compare ASSET propagation directly against known
    closed-form solutions for circular two-body motion.
    """

    def test_quarter_period_exact_solution(self):
        """
        Verify the analytical quarter-period circular-orbit solution.

        For mu = 1 and r = 1:

            T = 2*pi
            T/4 = pi/2

        Expected state:

            r = [0, 1, 0]
            v = [-1, 0, 0]
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        final_state = self.propagate(initial_state, math.pi / 2.0)

        expected_state = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])

        np.testing.assert_allclose(final_state, expected_state, atol=1.0e-12)

    def test_half_period_exact_solution(self):
        """
        Verify the analytical half-period circular-orbit solution.

        At T/2 = pi:

            r = [-1, 0, 0]
            v = [0, -1, 0]
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        final_state = self.propagate(initial_state, math.pi)

        expected_state = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])

        np.testing.assert_allclose(final_state, expected_state, atol=1.0e-12)

    def test_full_period_returns_to_initial_state(self):
        """
        Verify that one complete circular period returns to the
        initial position and velocity.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        period = 2.0 * math.pi

        final_state = self.propagate(initial_state, period)

        np.testing.assert_allclose(final_state, initial_state, atol=1.0e-12)

    def test_circular_radius_is_constant(self):
        """
        Verify that the radius remains constant throughout circular
        propagation.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        for duration in (math.pi / 4.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0, 2.0 * math.pi):
            final_state = self.propagate(initial_state, duration)
            self.assertAlmostEqual(self.orbital_radius(final_state), 1.0, places=12)

    def test_circular_speed_is_constant(self):
        """
        Verify that the circular-orbit speed remains constant.

        For mu = 1 and r = 1:

            v = sqrt(mu / r) = 1
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        for duration in (0.5, 1.0, 2.0, 4.0, 2.0 * math.pi):
            final_state = self.propagate(initial_state, duration)
            self.assertAlmostEqual(self.orbital_speed(final_state), 1.0, places=12)

    def test_circular_orbit_has_zero_eccentricity(self):
        """
        Verify that a circular orbit has zero eccentricity.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        final_state = self.propagate(initial_state, 3.0)

        eccentricity = self.orbital_eccentricity(final_state, self.mu)

        self.assertAlmostEqual(eccentricity, 0.0, places=12)


# ======================================================================
# 2. Conservation-law tests
# ======================================================================

class KeplerConservationTests(KeplerPropagatorTestBase):
    """
    Validate conservation laws of the Newtonian two-body problem.

    The primary invariants are:

        Specific mechanical energy
        Specific angular momentum
        Orbital eccentricity
    """

    def test_energy_and_angular_momentum_are_conserved(self):
        """
        Verify conservation of specific mechanical energy and angular
        momentum for a non-circular three-dimensional orbit.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.35, 0.10])
        final_state = self.propagate(initial_state, 10.0)

        initial_energy = self.specific_energy(initial_state, self.mu)
        final_energy = self.specific_energy(final_state, self.mu)

        initial_h = self.specific_angular_momentum(initial_state)
        final_h = self.specific_angular_momentum(final_state)

        self.assertAlmostEqual(final_energy, initial_energy)
        np.testing.assert_allclose(final_h, initial_h)

    def test_eccentricity_is_conserved(self):
        """
        Verify that orbital eccentricity remains constant.

        The eccentricity is determined by the conserved specific energy
        and angular momentum.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.20, 0.25])
        initial_eccentricity = self.orbital_eccentricity(initial_state, self.mu)

        for duration in (1.0, 3.0, 7.0, 15.0):
            final_state = self.propagate(initial_state, duration)
            final_eccentricity = self.orbital_eccentricity(final_state, self.mu)

            self.assertAlmostEqual(final_eccentricity, initial_eccentricity, places=11)

    def test_energy_and_angular_momentum_remain_constant_over_multiple_orbits(self):
        """
        Verify conservation of physical invariants over several orbital
        periods.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        initial_energy = self.specific_energy(initial_state, self.mu)
        initial_h = self.specific_angular_momentum(initial_state)

        period = 2.0 * math.pi
        final_state = self.propagate(initial_state, 10.0 * period)

        final_energy = self.specific_energy(final_state, self.mu)
        final_h = self.specific_angular_momentum(final_state)

        self.assertAlmostEqual(final_energy, initial_energy, places=12)
        np.testing.assert_allclose(final_h, initial_h, atol=1.0e-11)


# ======================================================================
# 3. Orbital geometry tests
# ======================================================================

class KeplerOrbitalGeometryTests(KeplerPropagatorTestBase):
    """
    Validate preservation of orbital geometry.

    These tests verify that the orbital plane and associated angular
    momentum direction remain fixed during propagation.
    """

    def test_orbital_plane_is_conserved(self):
        """
        Verify that the orientation of the orbital plane remains fixed.

        The orbital plane normal is:

            h = r x v
        """
        initial_state = np.array([1.0, 0.2, 0.4, -0.15, 1.1, 0.35])
        initial_h = self.specific_angular_momentum(initial_state)

        final_state = self.propagate(initial_state, 12.0)
        final_h = self.specific_angular_momentum(final_state)

        initial_normal = initial_h / np.linalg.norm(initial_h)
        final_normal = final_h / np.linalg.norm(final_h)

        np.testing.assert_allclose(final_normal, initial_normal, atol=1.0e-11)

    def test_three_dimensional_motion_remains_in_initial_orbital_plane(self):
        """
        Verify that a three-dimensional trajectory remains in its
        initial orbital plane.

        The scalar triple product:

            h_initial dot r_final

        must remain zero.
        """
        initial_state = np.array([1.0, 0.2, 0.4, -0.15, 1.1, 0.35])
        initial_h = self.specific_angular_momentum(initial_state)

        final_state = self.propagate(initial_state, 8.0)
        final_position = final_state[:3]

        plane_error = np.dot(initial_h, final_position)

        self.assertAlmostEqual(plane_error, 0.0, places=10)


# ======================================================================
# 4. Keplerian scaling and orbital-period tests
# ======================================================================

class KeplerOrbitalScalingTests(KeplerPropagatorTestBase):
    """
    Validate analytical relationships between orbital radius, velocity,
    and period.

    These tests specifically examine Keplerian scaling laws.
    """

    def test_orbital_period_follows_keplers_third_law(self):
        """
        Verify Kepler's third law for a unit circular orbit.

            T = 2*pi*sqrt(r^3 / mu)
        """
        radius = 1.0
        expected_period = 2.0 * math.pi * math.sqrt(radius**3 / self.mu)
        initial_state = np.array([radius, 0.0, 0.0, 0.0, 1.0, 0.0])

        final_state = self.propagate(initial_state, expected_period)

        np.testing.assert_allclose(final_state, initial_state, atol=1.0e-12)

    def test_different_circular_radius_has_correct_speed(self):
        """
        Verify the circular velocity relation:

            v = sqrt(mu / r)

        for a radius other than one.
        """
        radius = 4.0
        expected_speed = math.sqrt(self.mu / radius)
        initial_state = np.array([radius, 0.0, 0.0, 0.0, expected_speed, 0.0])

        final_state = self.propagate(initial_state, 1.0)

        self.assertAlmostEqual(self.orbital_radius(final_state), radius, places=11)
        self.assertAlmostEqual(self.orbital_speed(final_state), expected_speed, places=11)

    def test_different_circular_radius_has_expected_period(self):
        """
        Verify Kepler's third law for a circular orbit at r = 4.

        With mu = 1:

            T = 2*pi*sqrt(4^3)
              = 16*pi
        """
        radius = 4.0
        circular_speed = math.sqrt(self.mu / radius)
        expected_period = 2.0 * math.pi * math.sqrt(radius**3 / self.mu)
        initial_state = np.array([radius, 0.0, 0.0, 0.0, circular_speed, 0.0])

        final_state = self.propagate(initial_state, expected_period)

        np.testing.assert_allclose(final_state, initial_state, atol=1.0e-11)


# ======================================================================
# 5. Orbit classification tests
# ======================================================================

class KeplerOrbitClassificationTests(KeplerPropagatorTestBase):
    """
    Validate classification of two-body trajectories using specific
    mechanical energy.

    The energy classification is:

        E < 0  -> bound elliptical orbit
        E = 0  -> parabolic escape trajectory
        E > 0  -> hyperbolic trajectory
    """

    def test_bound_orbit_has_negative_specific_energy(self):
        """
        Verify that a bound elliptical orbit has negative energy.
        """
        state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        energy = self.specific_energy(state, self.mu)

        self.assertLess(energy, 0.0)

    def test_escape_velocity_has_zero_specific_energy(self):
        """
        Verify that escape velocity produces zero specific energy.

        At r = 1:

            v_escape = sqrt(2*mu)
        """
        escape_speed = math.sqrt(2.0 * self.mu)
        state = np.array([1.0, 0.0, 0.0, 0.0, escape_speed, 0.0])
        energy = self.specific_energy(state, self.mu)

        self.assertAlmostEqual(energy, 0.0, places=14)

    def test_hyperbolic_orbit_has_positive_specific_energy(self):
        """
        Verify that a state above escape velocity has positive energy.
        """
        hyperbolic_speed = math.sqrt(2.0 * self.mu) + 0.5
        state = np.array([1.0, 0.0, 0.0, 0.0, hyperbolic_speed, 0.0])
        energy = self.specific_energy(state, self.mu)

        self.assertGreater(energy, 0.0)


# ======================================================================
# 6. Propagation consistency and reversibility
# ======================================================================

class KeplerPropagationConsistencyTests(KeplerPropagatorTestBase):
    """
    Validate numerical consistency properties of the propagator.

    These tests do not depend on a particular orbit being circular.
    Instead, they verify fundamental properties of autonomous two-body
    propagation.
    """

    def test_forward_then_backward_propagation_recovers_initial_state(self):
        """
        Verify time reversibility of the Kepler propagator.
        """
        initial_state = np.array([1.2, -0.3, 0.1, 0.2, 0.8, -0.15])
        duration = 3.75

        forward_state = self.propagate(initial_state, duration)
        recovered_state = self.propagate(forward_state, -duration)

        np.testing.assert_allclose(recovered_state, initial_state, atol=1.0e-11)

    def test_negative_propagation_is_consistent_with_forward_motion(self):
        """
        Verify consistency between forward and backward propagation.
        """
        initial_state = np.array([1.1, -0.2, 0.3, 0.1, 0.9, -0.2])
        duration = 2.5

        backward_state = self.propagate(initial_state, -duration)
        forward_state = self.propagate(initial_state, duration)
        recovered_state = self.propagate(forward_state, -2.0 * duration)

        np.testing.assert_allclose(recovered_state, backward_state, atol=1.0e-11)

    def test_zero_duration_returns_initial_state(self):
        """
        Verify that zero-duration propagation leaves the state unchanged.
        """
        initial_state = np.array([1.5, -0.4, 0.7, 0.2, 0.8, -0.1])
        final_state = self.propagate(initial_state, 0.0)

        np.testing.assert_allclose(final_state, initial_state, atol=1.0e-13)

    def test_propagation_is_consistent_when_split_into_intervals(self):
        """
        Verify the semigroup property of autonomous propagation.

        Propagation over:

            dt1 + dt2

        should match:

            propagate(dt1)
            then propagate(dt2)
        """
        initial_state = np.array([1.2, 0.1, -0.2, -0.1, 0.85, 0.25])
        dt1 = 2.25
        dt2 = 3.75

        direct_state = self.propagate(initial_state, dt1 + dt2)
        intermediate_state = self.propagate(initial_state, dt1)
        split_state = self.propagate(intermediate_state, dt2)

        np.testing.assert_allclose(split_state, direct_state, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()