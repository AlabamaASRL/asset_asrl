"""
Physics regression tests for two-body and CR3BP reference frames.

This test module validates the physical relationships used to construct the
``TwoBodyFrame`` and ``CR3BPFrame`` reference frames. The tests intentionally
focus on physics invariants, nondimensionalization identities, and known
properties of the Earth-Moon and Sun-Earth systems.

The regression tests cover:

TwoBodyFrame
-------------
* Characteristic length, time, velocity, and acceleration scaling.
* Keplerian circular-orbit relationships.
* Escape velocity and specific orbital energy.
* Nondimensional gravitational parameter normalization.
* Scaling behavior when the characteristic length changes.
* Solar-orbit characteristic scales at one astronomical unit.

CR3BPFrame
----------
* Earth-Moon mass fraction and characteristic scaling.
* Barycentric primary positions.
* Normalized separation between the primary bodies.
* Collinear Lagrange-point ordering.
* Triangular Lagrange-point geometry and symmetry.
* Rotating-frame equilibrium force balance.
* Nondimensional mean motion normalization.
* Effective potential and Jacobi constant symmetry.
* Earth-Moon orbital-period consistency.

References
----------
[1] Bate, R. R., Mueller, D. D., and White, J. E.,
    Fundamentals of Astrodynamics, Dover Publications, 1971.

[2] Szebehely, V.,
    Theory of Orbits: The Restricted Problem of Three Bodies,
    Academic Press, 1967.

[3] Koon, W. S., Lo, M. W., Marsden, J. E., and Ross, S. D.,
    Dynamical Systems, the Three-Body Problem and Space Mission Design,
    2008.

[4] NASA Jet Propulsion Laboratory,
    Solar System Dynamics and planetary physical constants.

"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Test module loading
# ---------------------------------------------------------------------------

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASTRO_ROOT = PACKAGE_ROOT / "Astro"


def load_module(name: str, path: pathlib.Path):
    """Load a Python module directly from a repository-relative file path.

    Parameters
    ----------
    name : str
        Temporary module name used when registering the module with
        ``sys.modules``.
    path : pathlib.Path
        Path to the Python source file to import.

    Returns
    -------
    module
        The loaded Python module.

    Notes
    -----
    Direct file loading allows the tests to execute against the source tree
    without requiring the complete package and all optional dependencies to be
    installed.
    """
    spec = importlib.util.spec_from_file_location(name, path)

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Optional ASSET dependency stub
# ---------------------------------------------------------------------------

# Frame construction and Lagrange-point calculations are NumPy-only. The
# production package imports ASSET, but these physics regression tests do not
# require any ASSET functionality. A minimal placeholder therefore permits the
# frame modules to load in a lightweight test environment.
asset_stub = types.ModuleType("asset")

asset_stub.VectorFunctions = types.SimpleNamespace(
    Arguments=None,
)

asset_stub.OptimalControl = types.SimpleNamespace()

sys.modules.setdefault("asset", asset_stub)


# ---------------------------------------------------------------------------
# Load constants and frame implementations
# ---------------------------------------------------------------------------

constants = load_module(
    "orbital_frame_constants",
    ASTRO_ROOT / "Constants.py",
)

two_body_module = load_module(
    "two_body_frame",
    ASTRO_ROOT / "Extensions" / "TwoBodyFrame.py",
)

cr3bp_module = load_module(
    "cr3bp_frame",
    ASTRO_ROOT / "Extensions" / "CR3BPFrame.py",
)

TwoBodyFrame = two_body_module.TwoBodyFrame
CR3BPFrame = cr3bp_module.CR3BPFrame


#%%
# ===========================================================================
# Two-body physics regression tests
# ===========================================================================

class TwoBodyFramePhysicsTests(unittest.TestCase):
    """
    Physics regression tests for two-body characteristic scaling.

    These tests verify that ``TwoBodyFrame`` satisfies the expected Keplerian
    relationships used to nondimensionalize a two-body gravitational system.

    The characteristic quantities are expected to satisfy:

        t* = sqrt(l*^3 / mu)

        v* = sqrt(mu / l*)

        a* = v* / t* = mu / l*^2

    where:

    * l* is the characteristic length,
    * t* is the characteristic time,
    * v* is the characteristic velocity,
    * a* is the characteristic acceleration, and
    * mu is the gravitational parameter.

    References
    ----------
    [1] Bate, Mueller, and White. Fundamentals of Astrodynamics.
        Two-body motion, Keplerian scaling, orbital energy, circular
        velocity, escape velocity, and orbital periods.
    """

    def test_earth_surface_scales_match_keplerian_relations(self):
        """
        Verify Earth-based characteristic scales satisfy Keplerian identities.

        A frame constructed using Earth's gravitational parameter and radius
        should produce internally consistent characteristic length, time,
        velocity, and acceleration scales.

        The expected relationships are:

            t* = sqrt(l*^3 / mu)

            v* = sqrt(mu / l*)

            a* = v* / t* = mu / l*^2

        Reference
        ---------
        [1] Bate, Mueller, and White, two-body nondimensionalization and
        Keplerian gravitational scaling.
        """
        frame = TwoBodyFrame(constants.MuEarth, constants.RadiusEarth,)

        self.assertEqual(frame.mu, 1)

        self.assertAlmostEqual(frame.tstar, np.sqrt(frame.lstar**3 / frame.P1mu))

        self.assertAlmostEqual(frame.vstar, np.sqrt(frame.P1mu / frame.lstar),)

        self.assertAlmostEqual(
            frame.astar,
            frame.vstar / frame.tstar,
        )

        self.assertAlmostEqual(
            frame.astar,
            frame.P1mu / frame.lstar**2,
        )

        self.assertAlmostEqual(
            frame.mustar,
            constants.MuEarth,
        )

    def test_characteristic_units_are_self_consistent(self):
        """
        Verify velocity and acceleration follow from length and time scales.

        Characteristic units should satisfy:

            v* = l* / t*

            a* = l* / t*^2

        These identities provide an independent consistency check on the
        nondimensionalization.

        Reference
        ---------
        [1] Bate, Mueller, and White, dimensional scaling relationships for
        two-body motion.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame.lstar / frame.tstar,
            frame.vstar,
        )

        self.assertAlmostEqual(
            frame.lstar / frame.tstar**2,
            frame.astar,
        )

    def test_nondimensional_gravitational_parameter_is_unity(self):
        """
        Verify the characteristic scaling normalizes gravity to mu = 1.

        The nondimensional gravitational parameter is:

            mu_nd = mu * t*^2 / l*^3

        A properly constructed two-body characteristic frame should normalize
        this quantity to one.

        Reference
        ---------
        [1] Bate, Mueller, and White, characteristic scaling of the
        two-body gravitational equations.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        nondimensional_mu = (
            frame.P1mu
            * frame.tstar**2
            / frame.lstar**3
        )

        self.assertAlmostEqual(
            nondimensional_mu,
            1.0,
        )

    def test_circular_orbit_has_unit_nondimensional_speed_and_period(self):
        """
        Verify a circular orbit at l* has unit speed and period 2*pi.

        At the characteristic radius:

            v_circular = sqrt(mu / l*)

        Since ``vstar`` uses the same expression, the nondimensional circular
        speed should be one.

        The dimensional circular period is:

            T = 2*pi*sqrt(l*^3 / mu)

        Therefore, the nondimensional period should be 2*pi.

        Reference
        ---------
        [1] Bate, Mueller, and White, circular-orbit velocity and period
        relations.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        circular_speed = np.sqrt(
            constants.MuEarth / constants.RadiusEarth,
        )

        circular_period = (
            2.0
            * np.pi
            * np.sqrt(
                constants.RadiusEarth**3
                / constants.MuEarth,
            )
        )

        self.assertAlmostEqual(
            circular_speed / frame.vstar,
            1.0,
        )

        self.assertAlmostEqual(
            circular_period / frame.tstar,
            2.0 * np.pi,
        )

    def test_escape_speed_is_sqrt_two_times_circular_speed(self):
        """
        Verify the standard two-body escape-speed relationship.

        For a circular orbit:

            v_circular = sqrt(mu / r)

        The local escape speed is:

            v_escape = sqrt(2*mu / r)

        Therefore:

            v_escape / v_circular = sqrt(2)

        Reference
        ---------
        [1] Bate, Mueller, and White, escape conditions in the two-body
        problem.
        """
        circular_speed = np.sqrt(
            constants.MuEarth / constants.RadiusEarth,
        )

        escape_speed = np.sqrt(
            2.0
            * constants.MuEarth
            / constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            escape_speed / circular_speed,
            np.sqrt(2.0),
        )

    def test_circular_orbit_specific_energy_matches_keplerian_value(self):
        """
        Verify circular-orbit specific energy matches the Kepler result.

        The specific mechanical energy is:

            epsilon = v^2 / 2 - mu / r

        For a circular orbit:

            epsilon = -mu / (2*r)

        When the orbit radius equals the characteristic length, the
        nondimensional specific energy should be -1/2.

        Reference
        ---------
        [1] Bate, Mueller, and White, specific orbital energy relations for
        Keplerian motion.
        """
        frame = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        radius = constants.RadiusEarth

        circular_speed = np.sqrt(
            constants.MuEarth / radius,
        )

        specific_energy = (
            0.5 * circular_speed**2
            - constants.MuEarth / radius
        )

        expected_energy = (
            -constants.MuEarth
            / (2.0 * radius)
        )

        self.assertAlmostEqual(
            specific_energy,
            expected_energy,
        )

        nondimensional_energy = (
            specific_energy
            / frame.vstar**2
        )

        self.assertAlmostEqual(
            nondimensional_energy,
            -0.5,
        )

    def test_escape_orbit_has_zero_specific_energy(self):
        """
        Verify escape velocity produces approximately zero specific energy.
    
        Escape velocity is defined as the speed at which the total specific
        mechanical energy reaches zero:
    
            v_escape = sqrt(2*mu/r)
    
        Substituting this velocity into:
    
            epsilon = v^2/2 - mu/r
    
        should produce zero within floating-point precision.
    
        Reference
        ---------
        [1] Bate, Mueller, and White, orbital energy and escape trajectories.
        """
        radius = constants.RadiusEarth
    
        escape_speed = np.sqrt(
            2.0
            * constants.MuEarth
            / radius,
        )
    
        specific_energy = (
            0.5 * escape_speed**2
            - constants.MuEarth / radius
        )
    
        self.assertAlmostEqual(
            specific_energy,
            0.0,
            delta=1.0e-8,
        )

    def test_characteristic_scales_change_consistently_with_length_scale(self):
        """
        Verify characteristic scales follow expected power laws.

        Holding the gravitational parameter fixed:

            t* proportional to l*^(3/2)

            v* proportional to l*^(-1/2)

            a* proportional to l*^(-2)

        Doubling the characteristic length therefore provides a regression
        test for all three scaling relationships.

        Reference
        ---------
        [1] Bate, Mueller, and White, Keplerian scaling laws.
        """
        frame1 = TwoBodyFrame(
            constants.MuEarth,
            constants.RadiusEarth,
        )

        frame2 = TwoBodyFrame(
            constants.MuEarth,
            2.0 * constants.RadiusEarth,
        )

        self.assertAlmostEqual(
            frame2.tstar / frame1.tstar,
            2.0**1.5,
        )

        self.assertAlmostEqual(
            frame2.vstar / frame1.vstar,
            1.0 / np.sqrt(2.0),
        )

        self.assertAlmostEqual(
            frame2.astar / frame1.astar,
            1.0 / 4.0,
        )

    def test_one_au_solar_orbit_has_about_one_year_period(self):
        """
        Verify a one-AU solar orbit has approximately a one-year period.

        The characteristic period for a frame constructed using the Sun's
        gravitational parameter and one astronomical unit should produce a
        full orbital period close to one calendar year.

        Reference
        ---------
        [1] Bate, Mueller, and White, Kepler's third law.

        [4] NASA JPL Solar System Dynamics, astronomical and heliocentric
        orbital scales.
        """
        frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )

        orbital_period_days = (
            2.0
            * np.pi
            * frame.tstar
            / constants.day
        )

        self.assertGreater(
            orbital_period_days,
            365.0,
        )

        self.assertLess(
            orbital_period_days,
            366.0,
        )

    def test_solar_characteristic_velocity_matches_earth_orbital_speed_scale(self):
        """
        Verify the one-AU solar velocity scale is near Earth's orbital speed.
    
        Earth's average heliocentric orbital speed is approximately:
    
            29.8 km/s
    
        When the frame constants use SI units, this corresponds to approximately:
    
            29,800 m/s
    
        The characteristic velocity for a Sun-centered frame using one AU as
        the characteristic length should therefore fall near this value.
    
        Reference
        ---------
        [1] Bate, Mueller, and White, circular-orbit velocity.
    
        [4] NASA JPL Solar System Dynamics, Earth heliocentric orbit scale.
        """
        frame = TwoBodyFrame(
            constants.MuSun,
            constants.AU,
        )
    
        # Constants are expressed in SI units, so velocity is in m/s.
        earth_orbital_speed_ms = frame.vstar
    
        self.assertGreater(
            earth_orbital_speed_ms,
            29_000.0,
        )
    
        self.assertLess(
            earth_orbital_speed_ms,
            31_000.0,
        )

#%%
# ===========================================================================
# CR3BP physics regression tests
# ===========================================================================

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
    timescale are well understood.

    References
    ----------
    [2] Szebehely, Theory of Orbits: The Restricted Problem of Three Bodies.

        Classical formulation of the circular restricted three-body problem,
        normalized primary coordinates, Lagrange points, effective potential,
        and Jacobi integral.

    [3] Koon, Lo, Marsden, and Ross, Dynamical Systems, the Three-Body
        Problem and Space Mission Design.

        Modern treatment of CR3BP nondimensionalization, rotating-frame
        dynamics, and Lagrange-point geometry.
    """

    def setUp(self):
        """
        Construct a normalized Earth-Moon CR3BP frame for each test.
        """
        self.frame = CR3BPFrame(
            constants.MuEarth,
            constants.MuMoon,
            constants.LD,
        )

    def test_mass_fraction_and_dimensional_scales_are_consistent(self):
        """
        Verify CR3BP mass parameter and characteristic scales.

        The normalized mass fraction is:

            mu = mu_moon / (mu_earth + mu_moon)

        The characteristic gravitational parameter is:

            mu* = mu_earth + mu_moon

        The characteristic time is:

            t* = sqrt(l*^3 / mu*)

        Reference
        ---------
        [2] Szebehely, CR3BP nondimensionalization.

        [3] Koon et al., normalized CR3BP reference frames.
        """
        frame = self.frame

        self.assertAlmostEqual(
            frame.mu,
            constants.MuMoon
            / (
                constants.MuEarth
                + constants.MuMoon
            ),
        )

        self.assertAlmostEqual(
            frame.mustar,
            constants.MuEarth
            + constants.MuMoon,
        )

        self.assertAlmostEqual(
            frame.tstar,
            np.sqrt(
                constants.LD**3
                / frame.mustar,
            ),
        )

        self.assertAlmostEqual(
            frame.vstar,
            constants.LD / frame.tstar,
        )

        self.assertAlmostEqual(
            frame.astar,
            frame.vstar / frame.tstar,
        )

    def test_primary_positions_are_barycentric_and_one_unit_apart(self):
        """
        Verify primary positions have zero barycenter and unit separation.

        In normalized CR3BP coordinates, the mass-weighted barycenter must be
        located at the origin:

            m1 * P1 + m2 * P2 = 0

        The normalized distance between the primary bodies must equal one.

        Reference
        ---------
        [2] Szebehely, barycentric rotating coordinates for the CR3BP.
        """
        frame = self.frame

        barycenter = (
            constants.MuEarth * frame.P1
            + constants.MuMoon * frame.P2
        )

        np.testing.assert_allclose(
            barycenter,
            np.zeros(3),
            atol=1.0e-15,
        )

        self.assertAlmostEqual(
            np.linalg.norm(
                frame.P2 - frame.P1,
            ),
            1.0,
        )

    def test_primary_positions_match_expected_normalized_coordinates(self):
        """
        Verify normalized barycentric locations of the primaries.

        For normalized mass parameter ``mu``:

            P1 = [-mu, 0, 0]

            P2 = [1-mu, 0, 0]

        Reference
        ---------
        [2] Szebehely, normalized barycentric primary coordinates.
        """
        frame = self.frame

        np.testing.assert_allclose(
            frame.P1,
            np.array(
                [
                    -frame.mu,
                    0.0,
                    0.0,
                ]
            ),
            atol=1.0e-15,
        )

        np.testing.assert_allclose(
            frame.P2,
            np.array(
                [
                    1.0 - frame.mu,
                    0.0,
                    0.0,
                ]
            ),
            atol=1.0e-15,
        )

    def test_primary_distances_from_barycenter_match_mass_fraction(self):
        """
        Verify primary distances from the barycenter match CR3BP theory.

        In normalized barycentric coordinates:

            |P1| = mu

            |P2| = 1 - mu

        Reference
        ---------
        [2] Szebehely, barycentric coordinate normalization.
        """
        frame = self.frame

        self.assertAlmostEqual(
            np.linalg.norm(frame.P1),
            frame.mu,
        )

        self.assertAlmostEqual(
            np.linalg.norm(frame.P2),
            1.0 - frame.mu,
        )

    def test_triangular_lagrange_points_form_equilateral_triangles(self):
        """
        Verify L4 and L5 form equilateral triangles with the primaries.

        Both triangular Lagrange points must be exactly one normalized unit
        away from each primary:

            |L4 - P1| = |L4 - P2| = 1

            |L5 - P1| = |L5 - P2| = 1

        Their y coordinates have magnitude:

            sqrt(3) / 2

        Reference
        ---------
        [2] Szebehely, triangular Lagrange-point solutions.

        [3] Koon et al., L4 and L5 geometry in normalized CR3BP coordinates.
        """
        frame = self.frame

        for point in (
            frame.L4,
            frame.L5,
        ):
            self.assertAlmostEqual(
                np.linalg.norm(
                    point - frame.P1,
                ),
                1.0,
            )

            self.assertAlmostEqual(
                np.linalg.norm(
                    point - frame.P2,
                ),
                1.0,
            )

            self.assertAlmostEqual(
                abs(point[1]),
                np.sqrt(3.0) / 2.0,
            )

    def test_l4_and_l5_are_mirror_images_across_x_axis(self):
        """
        Verify L4 and L5 are symmetric about the x-axis.

        The triangular equilibrium points should share identical x and z
        coordinates while their y coordinates have equal magnitude and
        opposite signs.

        Reference
        ---------
        [2] Szebehely, symmetry of the triangular Lagrange points.
        """
        frame = self.frame

        self.assertAlmostEqual(
            frame.L4[0],
            frame.L5[0],
        )

        self.assertAlmostEqual(
            frame.L4[1],
            -frame.L5[1],
        )

        self.assertAlmostEqual(
            frame.L4[2],
            frame.L5[2],
        )

    def test_triangular_lagrange_points_have_expected_x_coordinate(self):
        """
        Verify L4 and L5 share the analytical CR3BP x coordinate.

        The triangular equilibrium points occur at:

            x = 1/2 - mu

        Reference
        ---------
        [2] Szebehely, analytical triangular Lagrange-point coordinates.
        """
        frame = self.frame

        expected_x = 0.5 - frame.mu

        self.assertAlmostEqual(
            frame.L4[0],
            expected_x,
        )

        self.assertAlmostEqual(
            frame.L5[0],
            expected_x,
        )

    def test_triangular_lagrange_points_have_expected_y_coordinates(self):
        """
        Verify the analytical y coordinates of L4 and L5.

        The normalized triangular Lagrange-point coordinates are:

            L4_y = +sqrt(3)/2

            L5_y = -sqrt(3)/2

        Reference
        ---------
        [2] Szebehely, analytical L4 and L5 solutions.
        """
        frame = self.frame

        expected_y = np.sqrt(3.0) / 2.0

        self.assertAlmostEqual(
            frame.L4[1],
            expected_y,
        )

        self.assertAlmostEqual(
            frame.L5[1],
            -expected_y,
        )

    def test_lagrange_points_are_planar(self):
        """
        Verify all classical Lagrange points lie in the xy plane.

        The Earth-Moon primary bodies and classical equilibrium points are
        constructed in the rotating xy plane, so every equilibrium point
        should have zero z coordinate.

        Reference
        ---------
        [2] Szebehely, planar CR3BP equilibrium solutions.
        """
        frame = self.frame

        for point in (
            frame.L1,
            frame.L2,
            frame.L3,
            frame.L4,
            frame.L5,
        ):
            self.assertAlmostEqual(
                point[2],
                0.0,
                places=14,
            )

    def test_collinear_lagrange_points_have_expected_ordering(self):
        """
        Verify the physical ordering of the collinear Lagrange points.

        Moving from negative to positive x:

            L3 < P1 < L1 < P2 < L2

        This ordering is a fundamental property of the Earth-Moon CR3BP.

        Reference
        ---------
        [2] Szebehely, collinear Lagrange-point solutions.
        """
        frame = self.frame

        self.assertLess(
            frame.L3[0],
            frame.P1[0],
        )

        self.assertLess(
            frame.P1[0],
            frame.L1[0],
        )

        self.assertLess(
            frame.L1[0],
            frame.P2[0],
        )

        self.assertLess(
            frame.P2[0],
            frame.L2[0],
        )

    def test_all_lagrange_points_satisfy_rotating_frame_force_balance(self):
        """
        Verify all Lagrange points satisfy equilibrium force balance.

        At a CR3BP equilibrium point, the gradient of the effective potential
        must vanish:

            grad(Omega) = 0

        This means the centrifugal acceleration and the gravitational
        accelerations from both primaries exactly cancel.

        Reference
        ---------
        [2] Szebehely, CR3BP equations of motion and equilibrium solutions.

        [3] Koon et al., rotating-frame equilibrium conditions.
        """
        mu = self.frame.mu

        def gradient_of_effective_potential(point):
            """
            Evaluate the normalized CR3BP effective-potential gradient.

            Reference
            ---------
            [2] Szebehely, normalized CR3BP equations of motion.
            """
            x, y, z = point

            r1 = np.sqrt(
                (x + mu)**2
                + y**2
                + z**2,
            )

            r2 = np.sqrt(
                (x - 1.0 + mu)**2
                + y**2
                + z**2,
            )

            return np.array(
                [
                    (
                        x
                        - (1.0 - mu)
                        * (x + mu)
                        / r1**3
                        - mu
                        * (x - 1.0 + mu)
                        / r2**3
                    ),
                    (
                        y
                        - (1.0 - mu)
                        * y
                        / r1**3
                        - mu
                        * y
                        / r2**3
                    ),
                    (
                        -(1.0 - mu)
                        * z
                        / r1**3
                        - mu
                        * z
                        / r2**3
                    ),
                ]
            )

        for point in (
            self.frame.L1,
            self.frame.L2,
            self.frame.L3,
            self.frame.L4,
            self.frame.L5,
        ):
            np.testing.assert_allclose(
                gradient_of_effective_potential(point),
                np.zeros(3),
                atol=1.0e-12,
            )

    def test_cr3bp_nondimensional_mean_motion_is_unity(self):
        """
        Verify characteristic time normalizes mean motion to one.

        The dimensional mean motion is:

            n = sqrt((mu1 + mu2) / l^3)

        The characteristic time should satisfy:

            n * t* = 1

        Therefore, one nondimensional time unit corresponds to one radian of
        rotation of the primary-body system.

        Reference
        ---------
        [2] Szebehely, CR3BP normalization.

        [3] Koon et al., nondimensional mean motion.
        """
        frame = self.frame

        mean_motion = np.sqrt(
            frame.mustar
            / frame.lstar**3,
        )

        self.assertAlmostEqual(
            mean_motion * frame.tstar,
            1.0,
        )

    def test_cr3bp_velocity_scale_matches_mean_motion_times_length(self):
        """
        Verify v* equals mean motion multiplied by the length scale.

        Since:

            n = 1 / t*

        the characteristic velocity can also be written as:

            v* = n*l*

        Reference
        ---------
        [2] Szebehely, normalized rotating-frame scales.
        """
        frame = self.frame

        mean_motion = np.sqrt(
            frame.mustar
            / frame.lstar**3,
        )

        self.assertAlmostEqual(
            frame.vstar,
            mean_motion * frame.lstar,
        )

    def test_earth_moon_mass_fraction_is_small_but_nonzero(self):
        """
        Verify the Earth-Moon mass parameter remains physically realistic.

        The Earth-Moon CR3BP mass parameter is approximately 0.01215. A
        bounded regression check detects significant changes to constants or
        mass-normalization logic without overfitting to floating-point digits.

        Reference
        ---------
        [4] NASA JPL planetary constants and Earth-Moon system parameters.
        """
        frame = self.frame

        self.assertGreater(
            frame.mu,
            0.012,
        )

        self.assertLess(
            frame.mu,
            0.013,
        )

    def test_l4_and_l5_have_identical_effective_potential(self):
        """
        Verify symmetry gives L4 and L5 identical effective potential.

        The normalized CR3BP effective potential is:

            Omega =
                1/2 * (x^2 + y^2)
                + (1-mu)/r1
                + mu/r2

        Because L4 and L5 are mirror images across the x-axis, the effective
        potential must have the same value at both points.

        Reference
        ---------
        [2] Szebehely, CR3BP effective potential.
        """
        frame = self.frame
        mu = frame.mu

        def effective_potential(point):
            """
            Compute the normalized CR3BP effective potential.

            Reference
            ---------
            [2] Szebehely, effective potential formulation.
            """
            x, y, z = point

            r1 = np.sqrt(
                (x + mu)**2
                + y**2
                + z**2,
            )

            r2 = np.sqrt(
                (x - 1.0 + mu)**2
                + y**2
                + z**2,
            )

            return (
                0.5 * (x**2 + y**2)
                + (1.0 - mu) / r1
                + mu / r2
            )

        self.assertAlmostEqual(
            effective_potential(frame.L4),
            effective_potential(frame.L5),
            places=14,
        )

    def test_l4_and_l5_have_equal_jacobi_constants(self):
        """
        Verify the Jacobi constant is identical at L4 and L5.

        For zero rotating-frame velocity:

            C = x^2 + y^2
                + 2(1-mu)/r1
                + 2mu/r2

        Symmetry requires the Jacobi constant to be identical at the two
        triangular equilibrium points.

        Reference
        ---------
        [2] Szebehely, Jacobi integral for the circular restricted
        three-body problem.

        [3] Koon et al., nondimensional CR3BP dynamics and Jacobi constants.
        """
        frame = self.frame
        mu = frame.mu

        def jacobi_constant(point):
            """
            Evaluate the zero-velocity normalized Jacobi constant.

            Reference
            ---------
            [2] Szebehely, Jacobi integral.
            """
            x, y, z = point

            r1 = np.sqrt(
                (x + mu)**2
                + y**2
                + z**2,
            )

            r2 = np.sqrt(
                (x - 1.0 + mu)**2
                + y**2
                + z**2,
            )

            return (
                x**2
                + y**2
                + 2.0 * (1.0 - mu) / r1
                + 2.0 * mu / r2
            )

        c_l4 = jacobi_constant(frame.L4)
        c_l5 = jacobi_constant(frame.L5)

        self.assertAlmostEqual(
            c_l4,
            c_l5,
            places=14,
        )

    def test_triangular_lagrange_jacobi_constant_matches_closed_form(self):
        """
        Verify the L4/L5 Jacobi constant matches its analytical value.

        At L4 and L5:

            r1 = r2 = 1

            x = 1/2 - mu

            y^2 = 3/4

        Therefore, for zero rotating-frame velocity:

            C = (1/2 - mu)^2 + 3/4 + 2

        Reference
        ---------
        [2] Szebehely, triangular equilibrium points and Jacobi integral.
        """
        frame = self.frame
        mu = frame.mu

        expected_c = (
            (0.5 - mu)**2
            + 0.75
            + 2.0
        )

        def jacobi_constant(point):
            """Evaluate the zero-velocity normalized Jacobi constant.

            Reference
            ---------
            [2] Szebehely, Jacobi integral.
            """
            x, y, z = point

            r1 = np.sqrt(
                (x + mu)**2
                + y**2
                + z**2,
            )

            r2 = np.sqrt(
                (x - 1.0 + mu)**2
                + y**2
                + z**2,
            )

            return (
                x**2
                + y**2
                + 2.0 * (1.0 - mu) / r1
                + 2.0 * mu / r2
            )

        self.assertAlmostEqual(
            jacobi_constant(frame.L4),
            expected_c,
            places=12,
        )

        self.assertAlmostEqual(
            jacobi_constant(frame.L5),
            expected_c,
            places=12,
        )

    def test_lagrange_points_are_finite_three_dimensional_vectors(self):
        """
        Verify every Lagrange point has a valid Cartesian position.

        Each classical equilibrium point should be represented as a finite
        three-dimensional Cartesian vector. This provides a lightweight
        regression check against numerical solver failures that could produce
        NaNs, infinities, or malformed vectors.

        Reference
        ---------
        [2] Szebehely, classical CR3BP equilibrium-point formulation.
        """
        frame = self.frame

        for point in (
            frame.L1,
            frame.L2,
            frame.L3,
            frame.L4,
            frame.L5,
        ):
            self.assertEqual(
                point.shape,
                (3,),
            )

            self.assertTrue(
                np.all(np.isfinite(point)),
            )

    def test_earth_moon_characteristic_period_matches_lunar_orbit_scale(self):
        """
        Verify the CR3BP period is near the physical Earth-Moon timescale.

        A complete nondimensional revolution requires:

            T = 2*pi*t*

        The resulting dimensional period should be near the Moon's sidereal
        orbital period of approximately 27.3 days.

        Reference
        ---------
        [2] Szebehely, CR3BP mean motion and characteristic time.

        [4] NASA JPL Earth-Moon orbital parameters.
        """
        synodic_period_days = (
            2.0
            * np.pi
            * self.frame.tstar
            / constants.day
        )

        self.assertGreater(
            synodic_period_days,
            27.0,
        )

        self.assertLess(
            synodic_period_days,
            28.0,
        )


if __name__ == "__main__":
    unittest.main()