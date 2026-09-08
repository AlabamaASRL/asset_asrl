# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import unittest

import asset as ast
import numpy as np


class KeplerPropagatorPhysicsTests(unittest.TestCase):
    """
    Regression tests for the physical behavior of ASSET's Kepler propagator.

    The tests validate the propagator against fundamental properties of the
    Newtonian two-body problem.

    The primary physical regression checks are:

    * Conservation of specific mechanical energy.
    * Conservation of specific angular momentum.
    * Exact circular-orbit propagation.
    * Time reversibility of the equations of motion.

    References
    ----------
    [1] Bate, R. R., Mueller, D. D., and White, J. E.,
        Fundamentals of Astrodynamics, Dover Publications, 1971.

    [2] Curtis, H. D.,
        Orbital Mechanics for Engineering Students, 4th Edition,
        Elsevier, 2020.

    [3] Battin, R. H.,
        An Introduction to the Mathematics and Methods of Astrodynamics,
        Revised Edition, AIAA Education Series, 1999.
    """

    mu = 1.0
    propagator = None

    @classmethod
    def setUpClass(cls):
        """
        Construct the ASSET Kepler propagator used by all tests.

        The gravitational parameter is set to:

            mu = 1

        to create a nondimensional two-body problem.

        For a unit circular orbit with:

            r = 1

        and:

            mu = 1

        the required circular velocity is:

            v = sqrt(mu / r) = 1

        The corresponding orbital period is:

            T = 2*pi

        These normalized values make it possible to derive exact expected
        states for several regression tests.
        """
        cls.propagator = ast.Astro.Kepler.KeplerPropagator(cls.mu).vf()

    @staticmethod
    def specific_energy(state, mu):
        """
        Return the specific mechanical energy of a Cartesian two-body state.

        For an ideal two-body gravitational system, the specific mechanical
        energy is:

            E = 1/2 * |v|^2 - mu / |r|

        where:

        * ``r`` is the three-dimensional position vector.
        * ``v`` is the three-dimensional velocity vector.
        * ``mu`` is the gravitational parameter.

        Specific mechanical energy is conserved during ideal two-body motion.
        Therefore, comparing the value before and after propagation provides a
        physical regression check on the Kepler propagator.

        Parameters
        ----------
        state : array_like
            Cartesian state vector:

                [x, y, z, vx, vy, vz]

        mu : float
            Gravitational parameter of the central body.

        Returns
        -------
        float
            Specific mechanical energy of the state.
        """
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        return 0.5 * np.dot(velocity, velocity) - mu / np.linalg.norm(position)

    @staticmethod
    def specific_angular_momentum(state):
        """
        Return the specific angular momentum vector of a Cartesian state.

        The specific angular momentum of a two-body orbit is:

            h = r x v

        where ``r`` is the position vector and ``v`` is the velocity vector.

        Because the gravitational acceleration is directed along the position
        vector, the gravitational force produces no torque about the central
        body:

            tau = r x F = 0

        Consequently, the specific angular momentum vector is conserved
        throughout ideal two-body motion.

        Parameters
        ----------
        state : array_like
            Cartesian state vector:

                [x, y, z, vx, vy, vz]

        Returns
        -------
        numpy.ndarray
            Three-dimensional specific angular momentum vector.
        """
        position = np.asarray(state[:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        return np.cross(position, velocity)

    def propagate(self, state, duration):
        """
        Propagate a Cartesian state forward or backward in time.

        ASSET's Kepler propagator expects the six Cartesian state variables
        followed by the requested propagation duration:

            [x, y, z, vx, vy, vz, dt]

        This helper constructs that seven-element input vector, evaluates the
        underlying ASSET ``VectorFunction``, and converts the result back to a
        NumPy array.

        A positive duration propagates the state forward in time, while a
        negative duration propagates it backward in time.

        Notes
        -----
        The physical behavior being tested follows from the autonomous
        Newtonian two-body equations of motion. The propagation direction is
        determined by the sign of the elapsed time.
        """
        initial_condition = np.concatenate((np.asarray(state, dtype=float), [duration]))
        return np.asarray(self.propagator.compute(initial_condition), dtype=float)

    def test_circular_orbit_advances_by_a_quarter_period(self):
        """
        Verify exact quarter-period motion for a unit circular orbit.

        For:

            mu = 1

        and a unit-radius circular orbit, the initial state:

            r = [1, 0, 0]

            v = [0, 1, 0]

        corresponds to counterclockwise circular motion in the x-y plane.

        The orbital period is:

            T = 2*pi

        Therefore, after one quarter of a period:

            T/4 = pi/2

        the spacecraft should be located at:

            r = [0, 1, 0]

        with velocity:

            v = [-1, 0, 0]

        This test verifies that the ASSET Kepler propagator reproduces this
        analytically known state to numerical precision.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        final_state = self.propagate(initial_state, math.pi / 2.0)

        np.testing.assert_allclose(
            final_state,
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            atol=1.0e-12)

    def test_two_body_energy_and_angular_momentum_are_conserved(self):
        """
        Verify conservation of energy and angular momentum.

        In an ideal two-body gravitational problem, the gravitational force is
        conservative and central. Consequently, both specific mechanical
        energy and specific angular momentum are constants of motion.

        This test propagates a non-circular three-dimensional state for a
        finite duration and compares the initial and final values of:

            E = 1/2 |v|^2 - mu/|r|

        and:

            h = r x v

        Agreement between the initial and final quantities verifies important
        physical invariants of the Kepler propagator and provides a stronger
        regression check than comparing only a single propagated state.
        """
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.35, 0.10])
        final_state = self.propagate(initial_state, 10.0)

        initial_energy = self.specific_energy(initial_state, self.mu)
        final_energy = self.specific_energy(final_state, self.mu)

        self.assertAlmostEqual(final_energy, initial_energy)

        np.testing.assert_allclose(
            self.specific_angular_momentum(final_state),
            self.specific_angular_momentum(initial_state))

    def test_forward_then_backward_propagation_recovers_initial_state(self):
        """
        Verify time reversibility of the Kepler propagator.

        The ideal two-body equations of motion are time reversible. If a state
        is propagated forward by a duration ``dt`` and the resulting state is
        then propagated backward by ``-dt``, the original state should be
        recovered.

        This test therefore performs:

            initial state
                |
                | +dt
                v
            propagated state
                |
                | -dt
                v
            recovered initial state

        Because the ASSET Kepler propagator uses a two-body propagation model,
        the recovered state should agree with the original state to high
        numerical precision.
        """
        initial_state = np.array([1.2, -0.3, 0.1, 0.2, 0.8, -0.15])
        duration = 3.75

        forward_state = self.propagate(initial_state, duration)
        recovered_state = self.propagate(forward_state, -duration)

        np.testing.assert_allclose(recovered_state, initial_state, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()