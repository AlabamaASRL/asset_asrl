# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest

import numpy as np
from asset_asrl.Astro import SpiceRead as spice_read

class FakeSpice:
    """Provide deterministic SPICE responses for regression tests."""

    def __init__(self):
        self.spkezr_calls = []
        self.pxform_calls = []
        self.sxform_calls = []

    def spkezr(self, body, et, frame, aberration, center):
        """Return a deterministic Cartesian state."""
        self.spkezr_calls.append((body, et, frame, aberration, center))
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 0.0

    def pxform(self, source_frame, destination_frame, et):
        """Return a deterministic position transformation."""
        self.pxform_calls.append((source_frame, destination_frame, et))
        return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    def sxform(self, source_frame, destination_frame, et):
        """Return a deterministic state transformation."""
        self.sxform_calls.append((source_frame, destination_frame, et))
        return np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


fake_spice = FakeSpice()
sys.modules["spiceypy"] = fake_spice

#%% SPICE ephemeris state tests

class SpiceReadEphemerisTests(unittest.TestCase):
    """Verify SPICE ephemeris state conversions and sampling."""

    def setUp(self):
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_ephemeris_state_converts_spice_units(self):
        """Verify SPICE km and km/s states are scaled correctly."""
        state = spice_read.GetEphemState("MOON", 2451546.0, LU=1000.0, TU=10.0)
        np.testing.assert_allclose(state, [1, 2, 3, 40, 50, 60])

    def test_ephemeris_state_converts_julian_date_to_et(self):
        """Verify one day after J2000 corresponds to 86400 seconds ET."""
        spice_read.GetEphemState("MOON", 2451546.0, LU=1000.0, TU=10.0)
        self.assertEqual(fake_spice.spkezr_calls[0][1], 86400.0)

    def test_ephemeris_state_uses_expected_spice_arguments(self):
        """Verify the ephemeris query uses the expected SPICE conventions."""
        spice_read.GetEphemState("MOON", 2451546.0, LU=1000.0, TU=10.0)
        self.assertEqual(fake_spice.spkezr_calls[0], ("MOON", 86400.0, "ECLIPJ2000", "NONE", "SOLAR SYSTEM BARYCENTER"))

    def test_ephemeris_trajectory_has_nondimensional_elapsed_time(self):
        """Verify trajectory samples use elapsed nondimensional time."""
        states = spice_read.GetEphemTraj2("EARTH", 2451545.0, 2451548.0, 3, LU=1000.0, TU=86400.0)
        np.testing.assert_allclose([state[6] for state in states], [0, 1, 2])

    def test_ephemeris_trajectory_converts_each_spice_epoch(self):
        """Verify trajectory samples use the expected SPICE epochs."""
        spice_read.GetEphemTraj2("EARTH", 2451545.0, 2451548.0, 3, LU=1000.0, TU=86400.0)
        self.assertEqual([call[1] for call in fake_spice.spkezr_calls], [0, 86400, 172800])

    def test_ephemeris_trajectory_scales_velocity(self):
        """Verify trajectory velocity components use the requested time scale."""
        states = spice_read.GetEphemTraj2("EARTH", 2451545.0, 2451548.0, 3, LU=1000.0, TU=86400.0)
        np.testing.assert_allclose(states[0][:6], [1, 2, 3, 345600, 432000, 518400])

#%% SPICE pole-vector tests

class SpiceReadPoleVectorTests(unittest.TestCase):
    """Verify body pole extraction from SPICE rotation matrices."""

    def setUp(self):
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_pole_vector_uses_rotation_matrix_third_column(self):
        """Verify the body-frame +Z axis is extracted from the third matrix column."""
        poles = spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)
        np.testing.assert_allclose(poles, [[1, 0, 0, 0], [1, 0, 0, 1]])

    def test_pole_vector_has_nondimensional_elapsed_time(self):
        """Verify pole-vector times are measured relative to the initial epoch."""
        poles = spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)
        np.testing.assert_allclose(poles[:, 3], [0, 1])

    def test_pole_vector_converts_julian_dates_to_et(self):
        """Verify pole-vector queries use the expected SPICE epochs."""
        spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)
        self.assertEqual([call[2] for call in fake_spice.pxform_calls], [0, 86400])

    def test_pole_vector_uses_expected_frames(self):
        """Verify pole-vector queries use the requested reference frames."""
        spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)
        self.assertEqual([call[:2] for call in fake_spice.pxform_calls], [("IAU_EARTH", "J2000"), ("IAU_EARTH", "J2000")])



#%% SPICE state-frame transformation tests

class SpiceReadFrameTransformTests(unittest.TestCase):
    """Verify six-dimensional SPICE state transformations."""

    def setUp(self):
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_state_frame_transform_applies_six_by_six_matrix(self):
        """Verify the SPICE six-by-six transformation is applied to the state."""
        state = spice_read.SpiceFrameTransform("J2000", "ECLIPJ2000", np.arange(1.0, 7.0), 2451545.5)
        np.testing.assert_allclose(state, [2, 6, 12, 20, 30, 42])

    def test_state_frame_transform_converts_julian_date_to_et(self):
        """Verify one-half day after J2000 corresponds to 43200 seconds ET."""
        spice_read.SpiceFrameTransform("J2000", "ECLIPJ2000", np.arange(1.0, 7.0), 2451545.5)
        self.assertEqual(fake_spice.sxform_calls[0][2], 43200.0)

    def test_state_frame_transform_uses_expected_frames(self):
        """Verify the requested source and destination frames are passed to SPICE."""
        spice_read.SpiceFrameTransform("J2000", "ECLIPJ2000", np.arange(1.0, 7.0), 2451545.5)
        self.assertEqual(fake_spice.sxform_calls[0][:2], ("J2000", "ECLIPJ2000"))



#%% SPICE convention tests

class SpiceReadConventionTests(unittest.TestCase):
    """Verify physical and mathematical conventions used by SpiceRead."""

    def setUp(self):
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_spice_position_scaling_uses_kilometers_to_meters(self):
        """Verify position scaling converts kilometers to meters before LU scaling."""
        state = spice_read.GetEphemState("EARTH", 2451545.0, LU=1000.0, TU=1.0)
        np.testing.assert_allclose(state[:3], [1, 2, 3])

    def test_spice_velocity_scaling_uses_time_and_length_scales(self):
        """Verify velocity scaling uses TU divided by LU."""
        state = spice_read.GetEphemState("EARTH", 2451545.0, LU=1000.0, TU=10.0)
        np.testing.assert_allclose(state[3:], [40, 50, 60])

    def test_spice_state_has_six_components(self):
        """Verify the returned ephemeris state contains position and velocity."""
        state = spice_read.GetEphemState("EARTH", 2451545.0, LU=1000.0, TU=10.0)
        self.assertEqual(len(state), 6)

    def test_spice_pole_vector_has_three_components_and_time(self):
        """Verify each pole-vector sample contains a three-vector and elapsed time."""
        poles = spice_read.PoleVector("IAU_EARTH", "J2000", 2451545.0, 2451547.0, 2, TU=86400.0)
        self.assertEqual(poles.shape, (2, 4))

    def test_spice_frame_transform_preserves_state_dimension(self):
        """Verify frame transformation returns a six-component state."""
        state = spice_read.SpiceFrameTransform("J2000", "ECLIPJ2000", np.arange(1.0, 7.0), 2451545.5)
        self.assertEqual(len(state), 6)

if __name__ == "__main__":
    unittest.main()