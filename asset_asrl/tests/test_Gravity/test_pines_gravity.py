# -*- coding: utf-8 -*-
"""
Unit tests for the compiled ASSET Pines gravity extension.

Tests:
    - Degree-zero central gravity
    - Degree-two J2 gravity
    - Zonal rotational symmetry
    - Acceleration direction
    - Finite outputs at degree 20
    - Deterministic repeated evaluation
"""

import unittest

import numpy as np
import asset_asrl
import asset as ast


# ============================================================
# Constants: km, km^3/s^2, km/s^2
# ============================================================

MU = 398600.4418
R_REF = 6378.1363
J2 = 1.08262668e-3

CENTRAL_RTOL = 1e-10
J2_RTOL = 1e-7
SYMMETRY_RTOL = 1e-10


# ============================================================
# Compiled extension adapter
# ============================================================

def make_pines(degree, C=None, S=None):
    """Construct the actual compiled ASSET Pines gravity VF."""
    if C is None:
        C = {(0, 0): 1.0}
    if S is None:
        S = {}
    try:
        model = ast.Extensions.PinesGravity(MU, R_REF, C, S, degree)
        return model.vf()
    except:
        print("Pines Model could not be initialized... skipping tests")
        return


def evaluate_acceleration(vf, position):
    """
    Evaluate the compiled VectorFunction.

    Try three-component position and six-component state inputs.
    """
    position = np.asarray(position, dtype=float).reshape(3)
    state6 = np.concatenate((position, np.zeros(3)))

    errors = []

    for state in (position, state6):
        try:
            if hasattr(vf, "compute"):
                result = vf.compute(state)
            else:
                result = vf.eval(1, state)

            result = np.asarray(result, dtype=float).reshape(-1)

            if result.size == 3 and np.all(np.isfinite(result)):
                return result

            errors.append(
                f"Input size {state.size}: output shape {result.shape}, "
                f"values {result}"
            )

        except Exception as exc:
            errors.append(f"Input size {state.size}: {exc}")

    raise RuntimeError("Could not evaluate Pines acceleration:\n" + "\n".join(errors))


# ============================================================
# Independent analytical references
# ============================================================

def central_gravity(position):
    r = np.asarray(position, dtype=float)
    radius = np.linalg.norm(r)
    return -MU * r / radius**3


def j2_gravity(position):
    """
    Analytical central + J2 acceleration.

    Assumes fully normalized C20 = -J2/sqrt(5).
    """
    x, y, z = np.asarray(position, dtype=float)

    r2 = x*x + y*y + z*z
    radius = np.sqrt(r2)
    invr3 = 1.0 / (r2 * radius)

    z2r2 = z*z / r2
    q = 1.5 * J2 * R_REF**2 / r2

    return np.array([
        -MU * x * invr3 * (1.0 - q * (5.0*z2r2 - 1.0)),
        -MU * y * invr3 * (1.0 - q * (5.0*z2r2 - 1.0)),
        -MU * z * invr3 * (1.0 - q * (5.0*z2r2 - 3.0)),
    ])


# ============================================================
# Unit tests
# ============================================================

class TestPinesGravity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Build each model once for the test class."""
        cls.degree0_vf = make_pines(degree=0)

        C = {
            (0, 0): 1.0,
            (2, 0): -J2 / np.sqrt(5.0),
        }
        S = {(2, 0): 0.0}

        cls.degree2_vf = make_pines(degree=2, C=C, S=S)
        
        # Degree 20, central-only coefficients.
        cls.degree20_vf = make_pines(degree=20)

    # --------------------------------------------------------
    # Degree zero: central gravity
    # --------------------------------------------------------

    def test_degree_zero_matches_central_gravity(self):
        positions = [
            ("Equatorial X", [7000.0, 0.0, 0.0]),
            ("Equatorial Y", [0.0, 7000.0, 0.0]),
            ("North pole", [0.0, 0.0, 7000.0]),
            ("Near north pole", [0.0, 0.1, 7000.0]),
            ("General 3D", [
                4219.5407659,
                1163.14807065,
                -4912.55385412,
            ]),
            ("Negative X", [-7000.0, 0.0, 0.0]),
        ]

        for name, position in positions:
            with self.subTest(position=name):
                actual = evaluate_acceleration(self.degree0_vf, position)
                expected = central_gravity(position)

                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=CENTRAL_RTOL,
                    atol=1e-14,
                )

    def test_degree_zero_equatorial_x_reference(self):
        actual = evaluate_acceleration(self.degree0_vf, [7000.0, 0.0, 0.0])

        expected = np.array([
            -8.134702893878e-3,
            0.0,
            0.0,
        ])

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=CENTRAL_RTOL,
            atol=1e-14,
        )

    def test_degree_zero_acceleration_points_toward_origin(self):
        positions = [
            [7000.0, 200.0, 100.0],
            [-4000.0, 3000.0, 500.0],
            [1000.0, -2000.0, 6000.0],
        ]

        for position in positions:
            with self.subTest(position=position):
                r = np.asarray(position, dtype=float)
                a = evaluate_acceleration(self.degree0_vf, r)

                # Acceleration must be parallel to -r.
                cross_product = np.cross(r, a)

                self.assertLess(np.linalg.norm(cross_product), 1e-8)
                self.assertLess(np.dot(r, a), 0.0)

    # --------------------------------------------------------
    # Degree two: J2
    # --------------------------------------------------------

    def test_degree_two_matches_analytical_j2(self):
        positions = [
            ("Equatorial X", [7000.0, 0.0, 0.0]),
            ("Equatorial Y", [0.0, 7000.0, 0.0]),
            ("North pole", [0.0, 0.0, 7000.0]),
            ("Near north pole", [0.0, 0.1, 7000.0]),
            ("Inclined", [5000.0, 3000.0, 4000.0]),
            ("General 3D", [
                4219.5407659,
                1163.14807065,
                -4912.55385412,
            ]),
        ]

        for name, position in positions:
            with self.subTest(position=name):
                actual = evaluate_acceleration(
                    self.degree2_vf, position
                )
                expected = j2_gravity(position)

                self.assertTrue(np.all(np.isfinite(actual)))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=J2_RTOL,
                    atol=1e-14,
                )

    def test_j2_changes_equatorial_radial_acceleration(self):
        position = [7000.0, 0.0, 0.0]
        central = evaluate_acceleration(self.degree0_vf, position)
        harmonic = evaluate_acceleration(self.degree2_vf, position)
        self.assertGreater(abs(harmonic[0] - central[0]), 1e-8,)

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    def test_zonal_model_is_symmetric_about_z_axis(self):
        r1 = np.array([7000.0, 0.0, 1000.0])
        r2 = np.array([0.0, 7000.0, 1000.0])

        a1 = evaluate_acceleration(self.degree2_vf, r1)
        a2 = evaluate_acceleration(self.degree2_vf, r2)

        # Rotate +90 degrees about Z:
        # (x, y, z) -> (-y, x, z)
        expected = np.array([
            -a1[1],
            a1[0],
            a1[2],
        ])

        np.testing.assert_allclose(
            a2,
            expected,
            rtol=SYMMETRY_RTOL,
            atol=1e-14,
        )

    # --------------------------------------------------------
    # Numerical robustness
    # --------------------------------------------------------

    def test_degree20_outputs_are_finite(self):
        positions = [
            [7000.0, 0.0, 0.0],
            [0.0, 0.0, 7000.0],
            [1.0e5, -2.0e5, 3.0e5],
            [-8000.0, 1000.0, -2000.0],
        ]

        for position in positions:
            with self.subTest(position=position):
                actual = evaluate_acceleration(self.degree20_vf, position)

                self.assertEqual(actual.shape, (3,))
                self.assertTrue(np.all(np.isfinite(actual)))

    def test_repeated_evaluation_is_deterministic(self):
        position = [
            4219.5407659,
            1163.14807065,
            -4912.55385412,
        ]

        first = evaluate_acceleration(self.degree2_vf, position)

        for _ in range(20):
            current = evaluate_acceleration(
                self.degree2_vf,
                position,
            )

            np.testing.assert_array_equal(current, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
