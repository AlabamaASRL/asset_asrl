"""Physical regression tests for the ASSET two-body Kepler propagator."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import asset as ast
except ModuleNotFoundError:
    ast = None


def specific_energy(state, mu):
    """
    Specific mechanical energy is conserved by the ideal two-body problem.
    
    E = 0.5 * v ^ 2 - mu / r
    """
    position = np.asarray(state[:3], dtype=float)
    velocity = np.asarray(state[3:6], dtype=float)
    return 0.5 * np.dot(velocity, velocity) - mu / np.linalg.norm(position)


def specific_angular_momentum(state):
    """
    Specific angular momentum is also conserved in a central gravitational field.
    
    L = r x mu * p
    """
    return np.cross(np.asarray(state[:3], dtype=float), np.asarray(state[3:6], dtype=float))


@unittest.skipIf(ast is None, "requires the compiled ASSET 'asset' extension")
class KeplerPropagatorPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mu = 1.0
        cls.propagator = ast.Astro.Kepler.KeplerPropagator(cls.mu).vf()

    def propagate(self, state, duration):
        # KeplerPropagator expects the six-state Cartesian vector followed by propagation time.
        initial_condition = np.concatenate((np.asarray(state, dtype=float), [duration]))
        return np.asarray(self.propagator.compute(initial_condition), dtype=float)

    def test_circular_orbit_advances_by_a_quarter_period(self):
        # A unit circular orbit rotates 90 degrees in pi/2 nondimensional time units.
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        final_state = self.propagate(initial_state, math.pi / 2.0)

        np.testing.assert_allclose(final_state, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], atol=1.0e-12)

    def test_two_body_energy_and_angular_momentum_are_conserved(self):
        initial_state = np.array([1.0, 0.0, 0.0, 0.0, 1.35, 0.10])
        final_state = self.propagate(initial_state, 10.0)

        self.assertAlmostEqual(specific_energy(final_state, self.mu), specific_energy(initial_state, self.mu), places=12)
        np.testing.assert_allclose(
            specific_angular_momentum(final_state),
            specific_angular_momentum(initial_state),
            atol=1.0e-12,
        )

    def test_forward_then_backward_propagation_recovers_initial_state(self):
        # Exact Kepler propagation is time reversible.
        initial_state = np.array([1.2, -0.3, 0.1, 0.2, 0.8, -0.15])
        duration = 3.75

        recovered_state = self.propagate(self.propagate(initial_state, duration), -duration)

        np.testing.assert_allclose(recovered_state, initial_state, atol=1.0e-11)


if __name__ == "__main__":
    unittest.main()
