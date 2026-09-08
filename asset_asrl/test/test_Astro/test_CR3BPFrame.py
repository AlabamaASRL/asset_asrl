# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
import numpy as np

from asset_asrl.Astro.Extensions.CR3BPFrame import CR3BPFrame
from asset_asrl.Astro import Constants as constants


class CR3BPFramePhysicsTests(unittest.TestCase):
    """
    Physics regression tests for the Earth-Moon CR3BP reference frame.

    The circular restricted three-body problem is normalized such that:

        * The total primary mass is one.
        * The separation between the primaries is one.
        * The mean motion is one.
        * The barycenter is located at the origin.

    The Earth-Moon system provides a physically meaningful regression case
    because the expected mass fraction, Lagrange-point geometry, and orbital
    timescale are well established.

    References
    ----------
    [1] Szebehely, Theory of Orbits: The Restricted Problem of Three Bodies.

    [2] Koon et al., Dynamical Systems, the Three-Body Problem and Space
        Mission Design.
    """

    def setUp(self):
        """Construct a normalized Earth-Moon CR3BP frame for each test."""
        self.frame = CR3BPFrame(constants.MuEarth, constants.MuMoon, constants.LD)

    def test_mass_fraction_and_dimensional_scales_are_consistent(self):
        """
        Verify CR3BP mass parameter and characteristic scales.

        The normalized mass fraction is:

            mu = mu_moon / (mu_earth + mu_moon)

        The characteristic gravitational parameter is:

            mu* = mu_earth + mu_moon

        The characteristic time is:

            t* = sqrt(l*^3 / mu*)
        """
        frame = self.frame

        self.assertAlmostEqual(frame.mu, constants.MuMoon / (constants.MuEarth + constants.MuMoon))
        self.assertAlmostEqual(frame.mustar, constants.MuEarth + constants.MuMoon)
        self.assertAlmostEqual(frame.tstar, np.sqrt(constants.LD**3 / frame.mustar))
        self.assertAlmostEqual(frame.vstar, constants.LD / frame.tstar)
        self.assertAlmostEqual(frame.astar, frame.vstar / frame.tstar)

    def test_primary_positions_are_barycentric_and_one_unit_apart(self):
        """
        Verify primary positions have zero barycenter and unit separation.

        In normalized CR3BP coordinates, the mass-weighted barycenter is:

            m1 * P1 + m2 * P2 = 0

        and the normalized distance between the primary bodies is one.
        """
        frame = self.frame

        barycenter = constants.MuEarth * frame.P1 + constants.MuMoon * frame.P2

        np.testing.assert_allclose(barycenter, np.zeros(3), atol=1.0e-15)
        self.assertAlmostEqual(np.linalg.norm(frame.P2 - frame.P1), 1.0)

    def test_primary_positions_match_expected_normalized_coordinates(self):
        """
        Verify normalized barycentric locations of the primaries.

        For normalized mass parameter ``mu``:

            P1 = [-mu, 0, 0]
            P2 = [1 - mu, 0, 0]
        """
        frame = self.frame

        np.testing.assert_allclose(frame.P1, np.array([-frame.mu, 0.0, 0.0]), atol=1.0e-15)
        np.testing.assert_allclose(frame.P2, np.array([1.0 - frame.mu, 0.0, 0.0]), atol=1.0e-15)

    def test_primary_distances_from_barycenter_match_mass_fraction(self):
        """
        Verify primary distances from the barycenter match CR3BP theory.

        In normalized barycentric coordinates:

            |P1| = mu
            |P2| = 1 - mu
        """
        frame = self.frame

        self.assertAlmostEqual(np.linalg.norm(frame.P1), frame.mu)
        self.assertAlmostEqual(np.linalg.norm(frame.P2), 1.0 - frame.mu)

    def test_triangular_lagrange_points_form_equilateral_triangles(self):
        """
        Verify L4 and L5 form equilateral triangles with the primaries.

        Both triangular Lagrange points must be one normalized unit away
        from each primary:

            |L4 - P1| = |L4 - P2| = 1
            |L5 - P1| = |L5 - P2| = 1

        Their y coordinates have magnitude sqrt(3) / 2.
        """
        frame = self.frame

        for point in (frame.L4, frame.L5):
            self.assertAlmostEqual(np.linalg.norm(point - frame.P1), 1.0)
            self.assertAlmostEqual(np.linalg.norm(point - frame.P2), 1.0)
            self.assertAlmostEqual(abs(point[1]), np.sqrt(3.0) / 2.0)

    def test_l4_and_l5_are_mirror_images_across_x_axis(self):
        """
        Verify L4 and L5 are symmetric about the x-axis.
        """
        frame = self.frame

        self.assertAlmostEqual(frame.L4[0], frame.L5[0])
        self.assertAlmostEqual(frame.L4[1], -frame.L5[1])
        self.assertAlmostEqual(frame.L4[2], frame.L5[2])

    def test_triangular_lagrange_points_have_expected_x_coordinate(self):
        """
        Verify L4 and L5 share the analytical CR3BP x coordinate.

        The triangular equilibrium points occur at:

            x = 1/2 - mu
        """
        frame = self.frame
        expected_x = 0.5 - frame.mu

        self.assertAlmostEqual(frame.L4[0], expected_x)
        self.assertAlmostEqual(frame.L5[0], expected_x)

    def test_triangular_lagrange_points_have_expected_y_coordinates(self):
        """
        Verify the analytical y coordinates of L4 and L5.

        The normalized coordinates are:

            L4_y = +sqrt(3) / 2
            L5_y = -sqrt(3) / 2
        """
        frame = self.frame
        expected_y = np.sqrt(3.0) / 2.0

        self.assertAlmostEqual(frame.L4[1], expected_y)
        self.assertAlmostEqual(frame.L5[1], -expected_y)

    def test_lagrange_points_are_planar(self):
        """
        Verify all classical Lagrange points lie in the xy plane.
        """
        frame = self.frame

        for point in (frame.L1, frame.L2, frame.L3, frame.L4, frame.L5):
            self.assertAlmostEqual(point[2], 0.0, places=14)

    def test_collinear_lagrange_points_have_expected_ordering(self):
        """
        Verify the physical ordering of the collinear Lagrange points.

        Moving from negative to positive x:

            L3 < P1 < L1 < P2 < L2
        """
        frame = self.frame

        self.assertLess(frame.L3[0], frame.P1[0])
        self.assertLess(frame.P1[0], frame.L1[0])
        self.assertLess(frame.L1[0], frame.P2[0])
        self.assertLess(frame.P2[0], frame.L2[0])

    def test_all_lagrange_points_satisfy_rotating_frame_force_balance(self):
        """
        Verify all Lagrange points satisfy equilibrium force balance.

        At a CR3BP equilibrium point, the gradient of the effective potential
        must vanish:

            grad(Omega) = 0

        This means the centrifugal acceleration and gravitational
        accelerations from both primaries exactly cancel.
        """
        mu = self.frame.mu

        def gradient_of_effective_potential(point):
            """Evaluate the normalized CR3BP effective-potential gradient."""
            x, y, z = point

            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

            return np.array([
                x - (1.0 - mu) * (x + mu) / r1**3 - mu * (x - 1.0 + mu) / r2**3,
                y - (1.0 - mu) * y / r1**3 - mu * y / r2**3,
                -(1.0 - mu) * z / r1**3 - mu * z / r2**3,
            ])

        for point in (self.frame.L1, self.frame.L2, self.frame.L3, self.frame.L4, self.frame.L5):
            np.testing.assert_allclose(gradient_of_effective_potential(point), np.zeros(3), atol=1.0e-12)

    def test_cr3bp_nondimensional_mean_motion_is_unity(self):
        """
        Verify characteristic time normalizes mean motion to one.

        The dimensional mean motion is:

            n = sqrt((mu1 + mu2) / l^3)

        The characteristic time satisfies:

            n * t* = 1
        """
        frame = self.frame

        mean_motion = np.sqrt(frame.mustar / frame.lstar**3)

        self.assertAlmostEqual(mean_motion * frame.tstar, 1.0)

    def test_cr3bp_velocity_scale_matches_mean_motion_times_length(self):
        """
        Verify v* equals mean motion multiplied by the length scale.

        Since:

            n = 1 / t*

        the characteristic velocity can also be written as:

            v* = n * l*
        """
        frame = self.frame

        mean_motion = np.sqrt(frame.mustar / frame.lstar**3)

        self.assertAlmostEqual(frame.vstar, mean_motion * frame.lstar)

    def test_earth_moon_mass_fraction_is_small_but_nonzero(self):
        """
        Verify the Earth-Moon mass parameter remains physically realistic.

        The Earth-Moon CR3BP mass parameter is approximately 0.01215.
        """
        frame = self.frame

        self.assertGreater(frame.mu, 0.012)
        self.assertLess(frame.mu, 0.013)

    def test_l4_and_l5_have_identical_effective_potential(self):
        """
        Verify symmetry gives L4 and L5 identical effective potential.

        The normalized CR3BP effective potential is:

            Omega =
                1/2 * (x^2 + y^2)
                + (1-mu) / r1
                + mu / r2
        """
        frame = self.frame
        mu = frame.mu

        def effective_potential(point):
            """Compute the normalized CR3BP effective potential."""
            x, y, z = point

            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

            return 0.5 * (x**2 + y**2) + (1.0 - mu) / r1 + mu / r2

        self.assertAlmostEqual(effective_potential(frame.L4), effective_potential(frame.L5), places=14)

    def test_l4_and_l5_have_equal_jacobi_constants(self):
        """
        Verify the Jacobi constant is identical at L4 and L5.

        For zero rotating-frame velocity:

            C = x^2 + y^2
                + 2(1-mu) / r1
                + 2mu / r2
        """
        frame = self.frame
        mu = frame.mu

        def jacobi_constant(point):
            """Evaluate the zero-velocity normalized Jacobi constant."""
            x, y, z = point

            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

            return x**2 + y**2 + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2

        c_l4 = jacobi_constant(frame.L4)
        c_l5 = jacobi_constant(frame.L5)

        self.assertAlmostEqual(c_l4, c_l5, places=14)

    def test_triangular_lagrange_jacobi_constant_matches_closed_form(self):
        """
        Verify the L4/L5 Jacobi constant matches its analytical value.

        At L4 and L5:

            r1 = r2 = 1
            x = 1/2 - mu
            y^2 = 3/4

        Therefore:

            C = (1/2 - mu)^2 + 3/4 + 2
        """
        frame = self.frame
        mu = frame.mu

        expected_c = (0.5 - mu)**2 + 0.75 + 2.0

        def jacobi_constant(point):
            """Evaluate the zero-velocity normalized Jacobi constant."""
            x, y, z = point

            r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
            r2 = np.sqrt((x - 1.0 + mu)**2 + y**2 + z**2)

            return x**2 + y**2 + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2

        self.assertAlmostEqual(jacobi_constant(frame.L4), expected_c, places=12)
        self.assertAlmostEqual(jacobi_constant(frame.L5), expected_c, places=12)

    def test_lagrange_points_are_finite_three_dimensional_vectors(self):
        """
        Verify every Lagrange point has a valid Cartesian position.

        Each classical equilibrium point should be represented as a finite
        three-dimensional Cartesian vector.
        """
        frame = self.frame

        for point in (frame.L1, frame.L2, frame.L3, frame.L4, frame.L5):
            self.assertEqual(point.shape, (3,))
            self.assertTrue(np.all(np.isfinite(point)))

    def test_earth_moon_characteristic_period_matches_lunar_orbit_scale(self):
        """
        Verify the CR3BP period is near the physical Earth-Moon timescale.

        A complete nondimensional revolution requires:

            T = 2*pi*t*

        The resulting dimensional period should be near the Moon's sidereal
        orbital period of approximately 27.3 days.
        """
        orbital_period_days = 2.0 * np.pi * self.frame.tstar / constants.day

        self.assertGreater(orbital_period_days, 27.0)
        self.assertLess(orbital_period_days, 28.0)


if __name__ == "__main__":
    unittest.main()