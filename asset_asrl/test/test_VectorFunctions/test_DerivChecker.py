"""Unit tests for the VectorFunctions finite-difference diagnostic helper."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys
import types
import unittest

import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = PACKAGE_ROOT / "VectorFunctions" / "Extensions" / "DerivChecker.py"


class FakePyVectorFunction:
    """Small stand-in that supplies exact finite-difference derivatives."""

    instances = []
    jacobians = (
        np.array([[2.0, 0.0], [0.0, 3.0]]),
        np.array([[4.0, 2.0], [0.0, 6.0]]),
    )

    def __init__(self, input_rows, output_rows, callback, jacobian_step, hessian_step):
        # Execute the callback so the test verifies the diagnostic's wiring, not just construction.
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.jacobian_step = jacobian_step
        self.hessian_step = hessian_step
        self.callback_value = callback(np.array([0.25, -0.5]))
        self.instance_number = len(self.instances)
        self.instances.append(self)

    def jacobian(self, _x):
        return self.jacobians[self.instance_number % 2]


asset_stub = types.ModuleType("asset")
asset_stub.VectorFunctions = types.SimpleNamespace(PyVectorFunction=FakePyVectorFunction)
sys.modules.setdefault("asset", asset_stub)

spec = importlib.util.spec_from_file_location("asset_asrl_deriv_checker", MODULE_PATH)
assert spec and spec.loader
deriv_checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deriv_checker)


class FakeFunction:
    def __init__(self):
        self.compute_inputs = []
        self.adjoint_inputs = []

    def IRows(self):
        return 2

    def ORows(self):
        return 2

    def compute(self, x):
        # This linear function has the exact Jacobian returned below.
        self.compute_inputs.append(x)
        return np.array([2.0 * x[0], 3.0 * x[1]])

    def jacobian(self, _x):
        return np.array([[2.0, 0.0], [0.0, 3.0]])

    def adjointgradient(self, x, l):
        # This gradient corresponds to the symmetric Hessian returned below.
        self.adjoint_inputs.append((x, l.copy()))
        return np.array([4.0 * x[0] + x[1], x[0] + 6.0 * x[1]])

    def adjointhessian(self, _x, _l):
        return np.array([[4.0, 1.0], [1.0, 6.0]])


class FDDerivCheckerTests(unittest.TestCase):
    def setUp(self):
        FakePyVectorFunction.instances.clear()

    def test_runs_all_step_sizes_with_expected_wrappers_and_unit_adjoint(self):
        # Each step size creates one value and one adjoint finite-difference wrapper.
        function = FakeFunction()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            deriv_checker.FDDerivChecker(function, np.array([0.25, -0.5]))

        self.assertEqual(len(FakePyVectorFunction.instances), 12)
        self.assertEqual(len(function.compute_inputs), 6)
        self.assertEqual(len(function.adjoint_inputs), 6)
        self.assertEqual(output.getvalue().count("Step Size:"), 6)

        expected_steps = [1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
        for index, step in enumerate(expected_steps):
            value_wrapper, adjoint_wrapper = FakePyVectorFunction.instances[2 * index : 2 * index + 2]
            self.assertEqual((value_wrapper.input_rows, value_wrapper.output_rows), (2, 2))
            self.assertEqual((adjoint_wrapper.input_rows, adjoint_wrapper.output_rows), (2, 2))
            self.assertEqual(value_wrapper.jacobian_step, step)
            self.assertEqual(value_wrapper.hessian_step, step)
            self.assertEqual(adjoint_wrapper.jacobian_step, step)
            self.assertEqual(adjoint_wrapper.hessian_step, step)
            np.testing.assert_array_equal(function.adjoint_inputs[index][1], np.ones(2))

    def test_reports_zero_error_when_analytic_and_mocked_finite_differences_match(self):
        # Matching analytic and finite-difference derivatives should report no residual.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            deriv_checker.FDDerivChecker(FakeFunction(), np.array([1.0, 2.0]))

        self.assertIn("Abs Max Jacobian Error:  0.0", output.getvalue())
        self.assertIn("Abs Max Hessian Error:  0.0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
