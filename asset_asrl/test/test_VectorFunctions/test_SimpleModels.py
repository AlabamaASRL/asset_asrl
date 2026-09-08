"""Regression tests using real ASSET VectorFunction and ODE primitives."""

from __future__ import annotations

import math
import unittest

import numpy as np

try:
    import asset as ast
except ModuleNotFoundError:
    ast = None


@unittest.skipIf(ast is None, "requires the compiled ASSET 'asset' extension")
class AssetSimpleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vf = ast.VectorFunctions
        cls.oc = ast.OptimalControl

    def test_quadratic_vector_function_values_and_derivatives(self):
        """
            This test constructs the analytical vector function
    
               f(x, y) = [x ^ 2 + 3y, xy]
    
           and evaluates it at
    
               (x, y) = (2, -1)
    
           The test verifies four quantities returned by ASSET's computeall()
           interface:
    
               * The vector-function value f(x)
               * The Jacobian df/dx
               * The gradient of the weighted/adjoint scalar function lambda^T f(x)
               * The Hessian of lambda^T f(x)
    
           The expected results are derived analytically so this test provides a
           regression check for both ASSET's symbolic graph construction and its
           automatic differentiation implementation.
       """
        x, y = self.vf.Arguments(2).tolist()
        function = self.vf.stack([x**2 + 3.0 * y, x * y])
        point = np.array([2.0, -1.0])
        weights = np.array([2.0, -1.0])

        value, jacobian, gradient, hessian = function.computeall(point, weights)

        np.testing.assert_allclose(value, [1.0, -2.0], atol=1e-14)
        np.testing.assert_allclose(jacobian, [[4.0, 3.0], [-1.0, 2.0]], atol=1e-14)
        np.testing.assert_allclose(gradient, [9.0, 4.0], atol=1e-14)
        np.testing.assert_allclose(hessian, [[4.0, -1.0], [-1.0, 0.0]], atol=1e-14)

    def test_simple_harmonic_oscillator_equations_of_motion(self):
        """
            The test constructs the unit-frequency simple harmonic oscillator
    
                xdot = v
                vdot = -x
    
            using ASSET's ODEArguments and ode_x.ode interfaces.
    
            The resulting ASSET ODE is evaluated at the state
    
                [x, v, t] = [3, -4, 0]
    
            The analytical equations require
    
                xdot = -4
                vdot = -3
    
            This test therefore verifies that ASSET correctly maps symbolic state
            variables into the resulting ODE VectorFunction and evaluates the
            equations of motion correctly.
        """
        args = self.oc.ODEArguments(2)
        position = args.XVar(0)
        velocity = args.XVar(1)
        oscillator = self.oc.ode_x.ode(self.vf.stack([velocity, -position]), 2)
 
        derivative = oscillator.vf().compute([3.0, -4.0, 0.0])

        # For x'' = -x, the derivative must be [velocity, -position].
        np.testing.assert_allclose(derivative, [-4.0, -3.0], atol=1e-14)

    def test_simple_harmonic_oscillator_returns_after_one_period(self):
        """
            A unit-frequency simple harmonic oscillator satisfies
    
                xdot = v
                vdot = -x
    
            Its analytical solution for the initial condition
    
                x(0) = 1
                v(0) = 0
    
            is
    
                x(t) = cos(t)
                v(t) = -sin(t)
    
            The oscillator has a period of
    
                T = 2*pi
    
            Consequently, the exact state at t = 2*pi is identical to the
            initial state:
    
                x(2*pi) = 1
                v(2*pi) = 0
    
            This test verifies that ASSET's ODE integrator reproduces this known
            analytical result within the specified numerical tolerance.
    
            The test also verifies that the integrator reaches the requested final
            independent-variable value of 2*pi
        """
        args = self.oc.ODEArguments(2)
        position = args.XVar(0)
        velocity = args.XVar(1)
        oscillator = self.oc.ode_x.ode(self.vf.stack([velocity, -position]), 2)
        integrator = oscillator.integrator(0.01)
        integrator.Adaptive = True
        integrator.setAbsTol(1.0e-12)

        final_state = integrator.integrate([1.0, 0.0, 0.0], 2.0 * math.pi)

        # The unit oscillator returns to its initial position and velocity after 2*pi.
        np.testing.assert_allclose(final_state[:2], [1.0, 0.0], atol=1.0e-8)
        self.assertAlmostEqual(final_state[2], 2.0 * math.pi, places=12)


if __name__ == "__main__":
    unittest.main()
