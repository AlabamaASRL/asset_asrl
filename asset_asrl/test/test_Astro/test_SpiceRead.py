"""Kernel-facing physics tests for SPICE ephemeris and frame utilities."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASTRO_ROOT = PACKAGE_ROOT / "Astro"


def load_module(name: str, path: pathlib.Path):
    """Load a module under its production name without the compiled extension."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeSpice:
    """Deterministic SPICE substitute that records kernel-style API calls."""

    def __init__(self):
        self.spkezr_calls = []
        self.pxform_calls = []
        self.sxform_calls = []

    def spkezr(self, body, et, frame, aberration, center):
        self.spkezr_calls.append((body, et, frame, aberration, center))
        # SPICE states are km and km/s; the production code applies the unit scaling.
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0.0

    def pxform(self, source_frame, destination_frame, et):
        self.pxform_calls.append((source_frame, destination_frame, et))
        # A proper 90-degree rotation; its third column is the transformed body pole.
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def sxform(self, source_frame, destination_frame, et):
        self.sxform_calls.append((source_frame, destination_frame, et))
        return np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


# SpiceRead imports these modules through the package namespace.  Create only
# the small package tree it needs, and replace spiceypy before loading it.
asset_asrl = types.ModuleType("asset_asrl")
astro = types.ModuleType("asset_asrl.Astro")
asset_asrl.Astro = astro
sys.modules.setdefault("asset_asrl", asset_asrl)
sys.modules.setdefault("asset_asrl.Astro", astro)
fake_spice = FakeSpice()
sys.modules["spiceypy"] = fake_spice
constants = load_module("asset_asrl.Astro.Constants", ASTRO_ROOT / "Constants.py")
date = load_module("asset_asrl.Astro.Date", ASTRO_ROOT / "Date.py")
astro.Constants = constants
astro.Date = date
spice_read = load_module("asset_asrl.Astro.SpiceRead", ASTRO_ROOT / "SpiceRead.py")


class SpiceReadPhysicsTests(unittest.TestCase):
    def setUp(self):
        # Each test observes only the calls made by its own kernel operation.
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_get_ephem_state_converts_spice_kilometres_to_scaled_si_units(self):
        state = spice_read.GetEphemState("MOON", 2451546.0, LU=1000.0, TU=10.0)

        # Position: km -> m / LU. Velocity: km/s -> m/s * TU / LU.
        np.testing.assert_allclose(state, [1.0, 2.0, 3.0, 40.0, 50.0, 60.0])
        self.assertEqual(
            fake_spice.spkezr_calls,
            [("MOON", 86400.0, "ECLIPJ2000", "NONE", "SOLAR SYSTEM BARYCENTER")],
        )

    def test_ephemeris_trajectory_has_nondimensional_elapsed_time(self):
        states = spice_read.GetEphemTraj2("EARTH", 2451545.0, 2451548.0, 3, LU=1000.0, TU=86400.0)

        # Samples span the start plus equally spaced interior epochs, measured from t0.
        np.testing.assert_allclose([state[6] for state in states], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(states[0][:6], [1.0, 2.0, 3.0, 345600.0, 432000.0, 518400.0])
        self.assertEqual([call[1] for call in fake_spice.spkezr_calls], [0.0, 86400.0, 172800.0])

    def test_pole_vector_uses_the_rotation_matrix_third_column(self):
        poles = spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)

        # The mocked rotation maps the +Z body pole onto the +X inertial axis.
        np.testing.assert_allclose(poles, [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]])
        self.assertEqual([call[2] for call in fake_spice.pxform_calls], [0.0, 86400.0])

    def test_state_frame_transform_applies_spice_six_by_six_matrix(self):
        transformed = spice_read.SpiceFrameTransform("J2000", "ECLIPJ2000", np.arange(1.0, 7.0), 2451545.5)

        np.testing.assert_allclose(transformed, [2.0, 6.0, 12.0, 20.0, 30.0, 42.0])
        self.assertEqual(fake_spice.sxform_calls, [("J2000", "ECLIPJ2000", 43200.0)])


if __name__ == "__main__":
    unittest.main()
