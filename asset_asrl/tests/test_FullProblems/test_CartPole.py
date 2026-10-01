# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
import numpy as np
import asset as ast


vf = ast.VectorFunctions
oc = ast.OptimalControl
Args = vf.Arguments


class CartPole(oc.ode_x_u.ode):

    def __init__(self, l, m1, m2, g):
        args = oc.ODEArguments(4, 1)

        q1 = args.XVar(0)
        q2 = args.XVar(1)
        q1d = args.XVar(2)
        q2d = args.XVar(3)

        q1, q2, q1d, q2d = args.XVec().tolist()

        u = args.UVar(0)

        q1dd = (
            l * m2 * vf.sin(q2) * (q2d**2)
            + u
            + m2 * g * vf.cos(q2) * vf.sin(q2)
        ) / (
            m1 + m2 * (1 - vf.cos(q2)**2)
        )

        q2dd = -1 * (
            l * m2 * vf.cos(q2) * vf.sin(q2) * (q2d**2)
            + u * vf.cos(q2)
            + (m1 * g + m2 * g) * vf.sin(q2)
        ) / (
            l * m1 + l * m2 * (1 - vf.cos(q2)**2)
        )

        ode = vf.stack([q1d, q2d, q1dd, q2dd])

        super().__init__(ode, 4, 1)


#%% CartPole ODE tests

class test_CartPole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m1 = 1.0
        cls.m2 = 0.3
        cls.l = 0.5
        cls.g = 9.81
        cls.ode = CartPole(cls.l, cls.m1, cls.m2, cls.g)

    def test_Construction(self):
        self.assertIsNotNone(self.ode, "CartPole ODE failed to construct")

    def test_StateDerivativeStructure(self):

        # q1dot = q1d
        # q2dot = q2d

        state = np.array([0.2, 0.4, 0.5, -0.3])
        control = np.array([1.0])
        result = np.asarray(self.ode.eval(state, control))

        self.assertEqual(result.shape, (4,), "CartPole ODE should return four state derivatives")

        np.testing.assert_allclose(result[0], state[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[1], state[3], rtol=1e-12, atol=1e-12)

    def test_ZeroVelocityZeroControl(self):

        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0])
        result = np.asarray(self.ode.eval(state, control))

        self.assertEqual(result.shape, (4,))

        # Both positions are stationary.
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)

        # At q2 = 0, the gravitational terms vanish.
        self.assertAlmostEqual(result[2], 0.0)
        self.assertAlmostEqual(result[3], 0.0)

    def test_ZeroAngleGravity(self):

        state = np.array([0.0, 0.0, 0.0, 0.0])
        control = np.array([0.0])
        result = np.asarray(self.ode.eval(state, control))

        self.assertAlmostEqual(result[2], 0.0, places=12)
        self.assertAlmostEqual(result[3], 0.0, places=12)

    def test_ControlSignSymmetry(self):

        state = np.array([0.0, 0.0, 0.0, 0.0])

        plus = np.asarray(self.ode.eval(state, np.array([1.0])))
        minus = np.asarray(self.ode.eval(state, np.array([-1.0])))

        # Control should reverse the acceleration response.
        np.testing.assert_allclose(plus[2], -minus[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(plus[3], -minus[3], rtol=1e-12, atol=1e-12)

    def test_GravityAccelerationReference(self):

        q2 = np.pi / 4

        state = np.array([0.0, q2, 0.0, 0.0])
        control = np.array([0.0])
        result = np.asarray(self.ode.eval(state, control))

        c = np.cos(q2)
        s = np.sin(q2)

        expected_q1dd = self.m2 * self.g * c * s / (self.m1 + self.m2 * (1 - c**2))

        expected_q2dd = -((self.m1 + self.m2) * self.g * s / (self.l * self.m1 + self.l * self.m2 * (1 - c**2)))

        np.testing.assert_allclose(result[2], expected_q1dd, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[3], expected_q2dd, rtol=1e-12, atol=1e-12)

    def test_ControlAccelerationReference(self):

        q2 = 0.0

        state = np.array([0.0, q2, 0.0, 0.0])
        control_value = 2.0
        result = np.asarray(self.ode.eval(state, np.array([control_value])))

        expected_q1dd = control_value / self.m1
        expected_q2dd = -(control_value / (self.l * self.m1))

        np.testing.assert_allclose(result[2], expected_q1dd, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[3], expected_q2dd, rtol=1e-12, atol=1e-12)

    def test_VelocityDependence(self):

        state = np.array([0.0, np.pi / 4, 1.0, 2.0])
        control = np.array([0.0])
        result = np.asarray(self.ode.eval(state, control))

        # Kinematic portion must always be the velocities.
        np.testing.assert_allclose(result[0], state[2], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(result[1], state[3], rtol=1e-12, atol=1e-12)

        # Accelerations should be finite.
        self.assertTrue(np.all(np.isfinite(result)), "CartPole produced non-finite derivatives")

    def test_AnglePeriodicity(self):

        state1 = np.array([0.0, 0.7, 0.2, -0.4])
        state2 = state1.copy()
        state2[1] += 2.0 * np.pi
        control = np.array([0.5])

        result1 = np.asarray(self.ode.eval(state1, control))
        result2 = np.asarray(self.ode.eval(state2, control))

        np.testing.assert_allclose(result1, result2, rtol=1e-12, atol=1e-12)

    def test_ZeroAngularVelocity(self):
        state = np.array([0.4, 0.6, 1.2, 0.0])
        control = np.array([0.0])
        result = np.asarray(self.ode.eval(state, control))

        # q2dot should still be exactly zero.
        self.assertAlmostEqual(result[1], 0.0, places=12)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_FiniteStateRegression(self):
        rng = np.random.default_rng(12345)

        for _ in range(25):

            state = np.array([
                rng.uniform(-2.0, 2.0),
                rng.uniform(-np.pi, np.pi),
                rng.uniform(-5.0, 5.0),
                rng.uniform(-5.0, 5.0)
            ])

            control = np.array([rng.uniform(-20.0, 20.0)])
            result = np.asarray(self.ode.eval(state, control))

            self.assertEqual(result.shape, (4,))
            self.assertTrue(np.all(np.isfinite(result)), "CartPole produced non-finite derivatives")


#%% Optimization regression tests

class test_CartPoleOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.FinalObj = 58.83219229674185
        cls.MaxObjError = 0.1
        cls.MaximumIters = 20

    def problem_impl(self, tmode, cmode, nsegs):

        m1 = 1.0
        m2 = 0.3
        l = 0.5
        g = 9.81

        umax = 20
        dmax = 2

        tf = 2
        d = 1

        ts = np.linspace(0, tf, 100)

        IG = [
            [d * t / tf, np.pi * t / tf, 0, 0, t, 0.0]
            for t in ts
        ]

        ode = CartPole(l, m1, m2, g)

        phase = ode.phase(tmode, IG, nsegs)

        phase.setControlMode(cmode)

        phase.addBoundaryValue("Front", range(0, 5), [0, 0, 0, 0, 0])
        phase.addBoundaryValue("Back", range(0, 5), [d, np.pi, 0, 0, tf])

        phase.addLUVarBound("Path", 5, -umax, umax, 1.0)
        phase.addLUVarBound("Path", 0, -dmax, dmax, 1.0)

        phase.addIntegralObjective(Args(1)[0]**2, [5])

        phase.optimizer.PrintLevel = 0

        Flag = phase.optimize()

        Obj = phase.optimizer.LastObjVal
        ObjError = abs(Obj - self.FinalObj)

        self.assertLess(phase.optimizer.LastIterNum, self.MaximumIters, "Optimizer iterations exceeded expected maximum")
        self.assertEqual(Flag, ast.Solvers.ConvergenceFlags.CONVERGED, "Problem did not converge")
        self.assertLess(ObjError, self.MaxObjError, "Final objective significantly differs from known answer")

    def test_FullProblem(self):

        tmodes = [
            "LGL3",
            "LGL5",
            "LGL7",
            "Trapezoidal",
            "CentralShooting"
        ]

        nsegs = [256, 128, 96, 256, 256]
        for tmode, nseg in zip(tmodes, nsegs):
            with self.subTest(TranscriptionMode=tmode):
                with self.subTest(cmode="HighestOrderSpline"):
                    self.problem_impl(tmode, "HighestOrderSpline", nseg)

                with self.subTest(cmode="BlockConstant"):
                    self.problem_impl(tmode, "BlockConstant", nseg)


if __name__ == "__main__":
    unittest.main(exit=False)