"""
Kernel-facing physics tests for SPICE ephemeris and frame utilities.

This module validates the physical conventions and unit transformations used by
the SPICE ephemeris and reference-frame utilities. The production code accesses
SPICE through ``spiceypy``; these tests replace the SPICE interface with a
deterministic substitute so that unit conversion, time conversion, ephemeris
sampling, and frame transformations can be tested without requiring external
kernel files.

The regression tests cover:

* Conversion of SPICE Cartesian states from kilometers and kilometers per
  second into application-defined dimensional or nondimensional units.
* Conversion of Julian Date values into SPICE ephemeris time (ET).
* Construction of ephemeris trajectories with elapsed nondimensional time.
* Extraction of a body's pole direction from a SPICE rotation matrix.
* Application of six-by-six state transformation matrices.
* Verification that production utilities call the expected SPICE APIs.

References
----------
[1] NASA Navigation and Ancillary Information Facility (NAIF),
    SPICE Toolkit Documentation.

[2] Acton, C. H.,
    "Ancillary Data Services of NASA's Navigation and Ancillary Information
    Facility," Planetary and Space Science, Vol. 44, No. 1, pp. 65-70, 1996.

[3] NASA NAIF,
    SPICE Required Reading: Time Required Reading (TIME.REQ).

[4] NASA NAIF,
    SPICE Required Reading: Frames Required Reading (FRAMES.REQ).

[5] NASA NAIF,
    SPICE Required Reading: SPK Required Reading (SPK.REQ).

Notes
-----
The tests intentionally do not validate actual planetary positions. Instead,
they validate the physical and mathematical conventions used by the application
when communicating with the SPICE toolkit. A deterministic ``FakeSpice``
implementation returns known values so that expected unit conversions and frame
transformations can be calculated exactly.

The SPICE toolkit convention for Cartesian ephemeris states returned by
``spkezr`` is:

    position : kilometers

    velocity : kilometers / second

The production utilities are responsible for converting those values into the
length and time units requested by the caller.

References
----------
[1] NASA NAIF SPICE Toolkit Documentation.

[5] NASA NAIF SPK Required Reading.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASTRO_ROOT = PACKAGE_ROOT / "Astro"


def load_module(name: str, path: pathlib.Path):
    """
    Load a module under its production package name without extensions.

    The SPICE utilities import several modules using their production package
    namespace. These tests load the source modules directly from the repository
    while preserving the package names expected by the production imports.

    Parameters
    ----------
    name : str
        Fully qualified module name used when registering the module in
        ``sys.modules``.

    path : pathlib.Path
        Repository-relative path to the Python source file.

    Returns
    -------
    module
        The imported module.

    Notes
    -----
    Loading modules directly from their source paths allows these tests to run
    independently of the compiled ASSET extension and installed package.
    """
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


class FakeSpice:
    """
    Deterministic SPICE substitute that records kernel-style API calls.

    This substitute implements the small subset of the ``spiceypy`` interface
    required by the production ``SpiceRead`` utilities:

    * ``spkezr`` for ephemeris state retrieval.
    * ``pxform`` for three-by-three position transformations.
    * ``sxform`` for six-by-six state transformations.

    Each method records its arguments so the tests can verify that the
    production utility converts time values and passes frame identifiers
    correctly.

    The returned values are intentionally simple and deterministic rather than
    physically realistic. Their purpose is to allow the tests to isolate and
    verify:

    * SPICE unit conversions.
    * Julian Date to ET conversion.
    * Frame transformation matrix application.
    * Extraction of geometric axes from rotation matrices.

    References
    ----------
    [1] NASA NAIF SPICE Toolkit Documentation.

    [4] NASA NAIF Frames Required Reading.

    [5] NASA NAIF SPK Required Reading.
    """

    def __init__(self):
        """
        Initialize call histories for the simulated SPICE interface.
        """
        self.spkezr_calls = []
        self.pxform_calls = []
        self.sxform_calls = []

    def spkezr(
        self,
        body,
        et,
        frame,
        aberration,
        center,
    ):
        """
        Return a deterministic Cartesian state and record the SPICE call.

        The real SPICE ``spkezr`` routine returns the geometric or apparent
        Cartesian state of a target body relative to an observer. The returned
        six-element state has the form:

            [x, y, z, vx, vy, vz]

        Under standard SPICE conventions:

            position units = kilometers

            velocity units = kilometers / second

        This test substitute returns:

            [1, 2, 3, 4, 5, 6]

        so that the production code's unit conversion can be checked exactly.

        Parameters
        ----------
        body : str
            SPICE target body identifier.

        et : float
            Ephemeris time in seconds past the J2000 epoch.

        frame : str
            SPICE reference frame in which the state is requested.

        aberration : str
            Aberration correction specification.

        center : str
            Observer or center body.

        Returns
        -------
        tuple
            ``(state, light_time)`` matching the ``spkezr`` interface.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ``spkezr``.

        [5] NASA NAIF SPK Required Reading, ephemeris state conventions.
        """
        self.spkezr_calls.append(
            (
                body,
                et,
                frame,
                aberration,
                center,
            )
        )

        # SPICE states use km and km/s. The production code applies any
        # requested dimensional or nondimensional unit scaling.
        return (
            np.array(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            ),
            0.0,
        )

    def pxform(
        self,
        source_frame,
        destination_frame,
        et,
    ):
        """
        Return a deterministic three-by-three SPICE rotation matrix.

        The real SPICE ``pxform`` routine returns a rotation matrix that
        transforms Cartesian position vectors between two reference frames.

        The test matrix is:

            [ 0  0  1 ]
            [ 0  1  0 ]
            [-1  0  0 ]

        This is a proper orthogonal rotation matrix. Its third column is:

            [1, 0, 0]

        Therefore, a body's positive local z-axis, commonly used to represent
        the body pole direction, transforms onto the positive x-axis of the
        destination frame.

        Parameters
        ----------
        source_frame : str
            Source SPICE reference frame.

        destination_frame : str
            Destination SPICE reference frame.

        et : float
            Ephemeris time in seconds past J2000.

        Returns
        -------
        numpy.ndarray
            Three-by-three rotation matrix.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ``pxform``.

        [4] NASA NAIF Frames Required Reading, rotation matrices and frame
        transformations.
        """
        self.pxform_calls.append(
            (
                source_frame,
                destination_frame,
                et,
            )
        )

        # A proper 90-degree rotation. Its third column is the transformed
        # body-frame +Z axis and therefore represents the transformed pole.
        return np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )

    def sxform(
        self,
        source_frame,
        destination_frame,
        et,
    ):
        """
        Return a deterministic six-by-six SPICE state transformation.

        The real SPICE ``sxform`` routine returns a six-by-six matrix used to
        transform Cartesian position and velocity states between time-dependent
        reference frames.

        A state transformation has the form:

            X_destination = SXFORM * X_source

        where:

            X = [x, y, z, vx, vy, vz]

        The deterministic diagonal matrix returned here makes the expected
        output easy to calculate exactly.

        Parameters
        ----------
        source_frame : str
            Source SPICE reference frame.

        destination_frame : str
            Destination SPICE reference frame.

        et : float
            Ephemeris time in seconds past J2000.

        Returns
        -------
        numpy.ndarray
            Six-by-six state transformation matrix.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ``sxform``.

        [4] NASA NAIF Frames Required Reading, state transformations.
        """
        self.sxform_calls.append(
            (
                source_frame,
                destination_frame,
                et,
            )
        )

        return np.diag(
            [
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ]
        )


# ---------------------------------------------------------------------------
# Minimal production package hierarchy
# ---------------------------------------------------------------------------

# SpiceRead imports these modules through the production package namespace.
# Create only the package hierarchy required by SpiceRead and replace the
# external spiceypy dependency with the deterministic FakeSpice implementation.

asset_asrl = types.ModuleType("asset_asrl")
astro = types.ModuleType("asset_asrl.Astro")

asset_asrl.Astro = astro

sys.modules.setdefault(
    "asset_asrl",
    asset_asrl,
)

sys.modules.setdefault(
    "asset_asrl.Astro",
    astro,
)


# ---------------------------------------------------------------------------
# Install deterministic SPICE substitute
# ---------------------------------------------------------------------------

fake_spice = FakeSpice()

sys.modules["spiceypy"] = fake_spice


# ---------------------------------------------------------------------------
# Load production Astro modules
# ---------------------------------------------------------------------------

constants = load_module(
    "asset_asrl.Astro.Constants",
    ASTRO_ROOT / "Constants.py",
)

date = load_module(
    "asset_asrl.Astro.Date",
    ASTRO_ROOT / "Date.py",
)

astro.Constants = constants
astro.Date = date

spice_read = load_module(
    "asset_asrl.Astro.SpiceRead",
    ASTRO_ROOT / "SpiceRead.py",
)


# ===========================================================================
# SPICE physics regression tests
# ===========================================================================

class SpiceReadPhysicsTests(unittest.TestCase):
    """
    Regression tests for SPICE ephemeris and frame utility conventions.

    These tests validate the mathematical and physical transformations applied
    by the production SPICE utility functions.

    The tests intentionally use deterministic mock SPICE responses rather than
    real kernel data. This isolates the application logic responsible for:

    * Time conversion.
    * Unit conversion.
    * Nondimensionalization.
    * Ephemeris sampling.
    * Rotation-matrix interpretation.
    * Six-dimensional state transformation.

    References
    ----------
    [1] NASA NAIF SPICE Toolkit Documentation.

    [3] NASA NAIF Time Required Reading.

    [4] NASA NAIF Frames Required Reading.

    [5] NASA NAIF SPK Required Reading.
    """

    def setUp(self):
        """
        Clear recorded SPICE calls before every regression test.

        Each test should observe only the SPICE operations performed by its own
        production utility call. Clearing the histories ensures that API-call
        verification is independent between tests.
        """
        fake_spice.spkezr_calls.clear()
        fake_spice.pxform_calls.clear()
        fake_spice.sxform_calls.clear()

    def test_get_ephem_state_converts_spice_kilometres_to_scaled_si_units(self):
        """
        Verify ephemeris states are converted from SPICE units correctly.

        SPICE ephemeris states use:

            position : kilometers

            velocity : kilometers / second

        The production utility applies the requested characteristic length
        scale ``LU`` and characteristic time scale ``TU``.

        Position scaling is:

            r_nd = r_spice * 1000 / LU

        Velocity scaling is:

            v_nd = v_spice * 1000 * TU / LU

        For the deterministic SPICE state:

            [1, 2, 3, 4, 5, 6]

        and:

            LU = 1000

            TU = 10

        the expected result is:

            [1, 2, 3, 40, 50, 60]

        The test also verifies that Julian Date 2451546.0 is converted to:

            ET = 86400 seconds

        because it occurs one day after J2000.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ephemeris state queries.

        [3] NASA NAIF Time Required Reading, Julian Date and ET.

        [5] NASA NAIF SPK Required Reading, km and km/s state units.
        """
        state = spice_read.GetEphemState(
            "MOON",
            2451546.0,
            LU=1000.0,
            TU=10.0,
        )

        # Position: km -> m / LU.
        # Velocity: km/s -> m/s * TU / LU.
        np.testing.assert_allclose(
            state,
            [
                1.0,
                2.0,
                3.0,
                40.0,
                50.0,
                60.0,
            ],
        )

        self.assertEqual(
            fake_spice.spkezr_calls,
            [
                (
                    "MOON",
                    86400.0,
                    "ECLIPJ2000",
                    "NONE",
                    "SOLAR SYSTEM BARYCENTER",
                )
            ],
        )

    def test_ephemeris_trajectory_has_nondimensional_elapsed_time(self):
        """
        Verify ephemeris trajectory samples use elapsed nondimensional time.

        The requested Julian Date interval spans:

            JD 2451545.0 to JD 2451548.0

        With three samples, the production utility should evaluate ephemeris
        states at:

            t = 0 days

            t = 1 day

            t = 2 days

        relative to the J2000 epoch.

        With:

            TU = 86400 seconds

        the elapsed nondimensional times should therefore be:

            [0, 1, 2]

        The test verifies both the returned elapsed time and the ephemeris time
        supplied to the underlying SPICE ``spkezr`` calls.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ephemeris state queries.

        [3] NASA NAIF Time Required Reading, ET and Julian Date conventions.

        [5] NASA NAIF SPK Required Reading, ephemeris state retrieval.
        """
        states = spice_read.GetEphemTraj2(
            "EARTH",
            2451545.0,
            2451548.0,
            3,
            LU=1000.0,
            TU=86400.0,
        )

        # Samples span the start plus equally spaced interior epochs,
        # measured relative to the initial epoch.
        np.testing.assert_allclose(
            [state[6] for state in states],
            [
                0.0,
                1.0,
                2.0,
            ],
        )

        np.testing.assert_allclose(
            states[0][:6],
            [
                1.0,
                2.0,
                3.0,
                345600.0,
                432000.0,
                518400.0,
            ],
        )

        self.assertEqual(
            [call[1] for call in fake_spice.spkezr_calls],
            [
                0.0,
                86400.0,
                172800.0,
            ],
        )

    def test_pole_vector_uses_the_rotation_matrix_third_column(self):
        """
        Verify the transformed body pole is extracted from the rotation matrix.

        A body's positive z-axis in its body-fixed frame is represented by:

            z_body = [0, 0, 1]

        Given a rotation matrix ``R`` that transforms vectors from the body
        frame to an inertial frame:

            z_inertial = R * z_body

        Multiplication by ``[0, 0, 1]`` selects the third column of ``R``.

        The deterministic test rotation has third column:

            [1, 0, 0]

        Therefore, the transformed body pole should align with the positive
        x-axis of the destination frame.

        The returned time values are measured relative to the first requested
        epoch and scaled by ``TU``.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ``pxform``.

        [3] NASA NAIF Time Required Reading.

        [4] NASA NAIF Frames Required Reading, rotation matrices and
        body-fixed frame transformations.
        """
        poles = spice_read.PoleVector(
            "IAU_EARTH",
            "J2000",
            2451545.0,
            2451547.0,
            2,
            TU=86400.0,
        )

        # The mocked rotation maps the body-frame +Z pole onto the
        # positive X-axis of the inertial frame.
        np.testing.assert_allclose(
            poles,
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        )

        self.assertEqual(
            [call[2] for call in fake_spice.pxform_calls],
            [
                0.0,
                86400.0,
            ],
        )

    def test_state_frame_transform_applies_spice_six_by_six_matrix(self):
        """
        Verify state frame transformations use the SPICE six-by-six matrix.

        A Cartesian state transformation is performed using:

            X_destination = S * X_source

        where:

            X = [x, y, z, vx, vy, vz]

        and ``S`` is the six-by-six state transformation matrix returned by
        SPICE ``sxform``.

        The deterministic transformation matrix used in this test is diagonal:

            diag([2, 3, 4, 5, 6, 7])

        Applying it to:

            [1, 2, 3, 4, 5, 6]

        produces:

            [2, 6, 12, 20, 30, 42]

        The test also verifies that Julian Date 2451545.5 is converted to:

            ET = 43200 seconds

        corresponding to one-half day after the J2000 epoch.

        References
        ----------
        [1] NASA NAIF SPICE Toolkit Documentation, ``sxform``.

        [3] NASA NAIF Time Required Reading, Julian Date to ET conversion.

        [4] NASA NAIF Frames Required Reading, six-by-six state
        transformations.
        """
        transformed = spice_read.SpiceFrameTransform(
            "J2000",
            "ECLIPJ2000",
            np.arange(1.0, 7.0),
            2451545.5,
        )

        np.testing.assert_allclose(
            transformed,
            [
                2.0,
                6.0,
                12.0,
                20.0,
                30.0,
                42.0,
            ],
        )

        self.assertEqual(
            fake_spice.sxform_calls,
            [
                (
                    "J2000",
                    "ECLIPJ2000",
                    43200.0,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()