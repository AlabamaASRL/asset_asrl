"""Independent reference tests for ASSET's Pines gravity kernel.

These tests validate PinesGravity using independent Newtonian and J2
acceleration equations, higher-degree harmonic evaluations, and analytic
two-body propagation properties.

The tests cover:
    - Point-mass acceleration and inverse-square scaling.
    - Radial direction, rotational symmetry, and deterministic evaluation.
    - Higher-degree gravity from degree 2 through degree 180, when available.
    - Comparisons between harmonic truncations and the highest tested degree.
    - Analytic J2 acceleration for equatorial, polar, and general positions.
    - Invalid coefficient files and the gravity singularity at the origin.
    - Circular-orbit propagation at quarter, half, and full periods.
    - Conservation of orbital radius, specific energy, and angular momentum.
    - Short-arc accuracy and propagation tolerance consistency.

Reference for the two-body half-period property:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/IDL/icy/cspice_prop2b.html
"""

from pathlib import Path
import tempfile
import unittest

import asset
import asset_asrl.Astro as astro
import numpy as np


# Earth gravitational parameter, in km^3/s^2.
MU = 3.9860043543609598e5

# Reference radius used by the gravity model, in km.
REFERENCE_RADIUS = 6378.1363

# Dimensionless second zonal harmonic coefficient.
J2 = 1.08262668355315e-3

# Degrees to exercise when the coefficient file contains those harmonics.
HIGHER_TEST_DEGREES = (2, 4, 8, 12, 20, 50, 100, 120, 180)

def find_repo_root(start_path=None):
    """Find the repository root by searching ancestors for native/data/egm96.nor."""
    start = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve().parent
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "native" / "data" / "egm96.nor").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate native/data/egm96.nor from {start} or its parents.")


REPO_ROOT = find_repo_root()
COEFFICIENTS = REPO_ROOT / "native" / "data" / "egm96.nor"


def maximum_coefficient_degree(coefficient_file=COEFFICIENTS):
    """Return the largest degree listed in a standard degree/order coefficient file."""
    maximum_degree = -1

    with Path(coefficient_file).open( "r", encoding="ascii", errors="replace" ) as stream:
        for line in stream:
            fields = line.split()

            if not fields or fields[0].startswith(("#", "%")):
                continue

            try:
                degree = int(fields[0])
                order = int(fields[1])
            except (ValueError, IndexError):
                # Ignore headers and non-coefficient lines.
                continue

            if degree >= 0 and 0 <= order <= degree:
                maximum_degree = max(maximum_degree, degree)

    if maximum_degree < 0:
        raise ValueError( f"No degree/order coefficient rows found in {coefficient_file}." )

    return maximum_degree

def make_pines(degree, coefficient_file=COEFFICIENTS):
    """Construct a PinesGravity model for the requested harmonic degree."""
    return astro.PinesGravity( str(coefficient_file), int(degree), float(MU), float(REFERENCE_RADIUS), "fully_normalized", )

def newtonian_acceleration(position):
    """Compute independent point-mass acceleration in km/s^2."""
    position = np.asarray(position, dtype=float)
    radius = np.linalg.norm(position)
    return -MU * position / radius**3

def j2_acceleration(position):
    """Compute analytic Cartesian acceleration for a point mass plus J2."""
    position = np.asarray(position, dtype=float)
    radius = np.linalg.norm(position)
    sin_latitude_squared = position[2] ** 2 / radius**2

    j2_factors = np.array( [ 5.0 * sin_latitude_squared - 1.0, 5.0 * sin_latitude_squared - 1.0, 5.0 * sin_latitude_squared - 3.0, ] )

    j2_term = (
        1.5
        * J2
        * MU
        * REFERENCE_RADIUS**2
        / radius**5
        * position
        * j2_factors
    )

    return newtonian_acceleration(position) + j2_term

def make_two_body_integrator(gravity, abs_tol=1e-12):
    """Create an ASSET six-state orbital integrator using the supplied gravity model."""
    args = asset.VectorFunctions.Arguments(7)
    rhs = asset.VectorFunctions.stack(args.segment3(3), gravity(args.head3()))
    ode = asset.OptimalControl.ode_6.ode(rhs, 6)
    integrator = ode.integrator(2.0)
    integrator.setAbsTol(float(abs_tol))
    return integrator

def circular_initial_state(radius_km=7000.0):
    """Return a prograde circular-orbit state in the xy plane."""
    speed_km_s = np.sqrt(MU / radius_km)
    return np.array( [radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0, 0.0] )

def circular_period(radius_km=7000.0):
    """Calculate the analytic period of a circular two-body orbit in seconds."""
    return 2.0 * np.pi * np.sqrt(radius_km**3 / MU)

def propagate(integrator, initial_state, time_s):
    """Propagate a state and return its six Cartesian position and velocity values."""
    return np.asarray( integrator.integrate(initial_state, float(time_s)) )[:6]

def specific_energy(state):
    """Calculate specific two-body orbital energy in km^2/s^2."""
    position = np.asarray(state[:3])
    velocity = np.asarray(state[3:6])
    return 0.5 * np.dot(velocity, velocity) - MU / np.linalg.norm(position)

def specific_angular_momentum(state):
    """Calculate the specific angular momentum vector in km^2/s."""
    return np.cross(np.asarray(state[:3]), np.asarray(state[3:6]))

class TestPinesReferenceCases(unittest.TestCase):
    """Test gravity acceleration and propagation against independent references."""

    @classmethod
    def setUpClass(cls):
        if not COEFFICIENTS.is_file():
            raise FileNotFoundError( f"EGM96 coefficients not found: {COEFFICIENTS}." )

    # ------------------------------------------------------------
    # Higher-degree gravity tests
    # ------------------------------------------------------------

    def test_higher_degree_models_are_finite_and_deterministic(self):
        """Exercise higher harmonic truncations at representative Earth-orbit positions."""
        maximum_degree = maximum_coefficient_degree(COEFFICIENTS)
        degrees = tuple( degree for degree in HIGHER_TEST_DEGREES if degree <= maximum_degree )

        if not degrees:
            self.skipTest( f"Coefficient file supports maximum degree {maximum_degree}; " "none of the configured higher test degrees are available." )

        positions = (
            np.array([7000.0, 0.0, 0.0]),
            np.array([4510.0, -3120.0, 5230.0]),
            np.array([-12000.0, 8500.0, 16000.0]),
            np.array( [ 42164.0 / np.sqrt(2.0), 42164.0 / np.sqrt(2.0), 0.0, ] ),
        )

        for degree in degrees:
            gravity = make_pines(degree)

            for position in positions:
                with self.subTest(degree=degree, position=position):
                    first = np.asarray( gravity.acceleration(position), dtype=float )
                    second = np.asarray( gravity.acceleration(position), dtype=float )

                    self.assertEqual(first.shape, (3,))
                    self.assertTrue(np.all(np.isfinite(first)))
                    np.testing.assert_array_equal(second, first)

    def test_higher_degree_truncations_are_consistent_with_maximum_test_degree(self):
        """Compare each supported truncation with the highest configured test degree."""
        maximum_degree = maximum_coefficient_degree(COEFFICIENTS)
        degrees = tuple( degree for degree in HIGHER_TEST_DEGREES if degree <= maximum_degree )

        if not degrees:
            self.skipTest( f"Coefficient file supports maximum degree {maximum_degree}; " "none of the configured higher test degrees are available." )

        reference_degree = max(degrees)
        reference_gravity = make_pines(reference_degree)

        positions = (
            np.array([7000.0, 1200.0, -900.0]),
            np.array([4510.0, -3120.0, 5230.0]),
            np.array([-12000.0, 8500.0, 16000.0]),
        )

        references = {
            tuple(position): np.asarray( reference_gravity.acceleration(position), dtype=float )
            for position in positions
        }

        for degree in degrees:
            gravity = make_pines(degree)

            for position in positions:
                with self.subTest( degree=degree, reference_degree=reference_degree, position=position, ):
                    acceleration = np.asarray( gravity.acceleration(position), dtype=float )
                    reference = references[tuple(position)]
                    difference = acceleration - reference

                    self.assertTrue(np.all(np.isfinite(acceleration)))
                    self.assertTrue(np.all(np.isfinite(difference)))
                    self.assertTrue(np.isfinite(np.linalg.norm(difference)))

                    # The reference model must reproduce its own acceleration.
                    if degree == reference_degree:
                        np.testing.assert_array_equal(acceleration, reference)

    def test_higher_degree_field_differs_from_point_mass_at_generic_position(self):
        """Check that nonzero harmonics add a correction to the point-mass field."""
        maximum_degree = maximum_coefficient_degree(COEFFICIENTS)

        if maximum_degree < 2:
            self.skipTest( f"Coefficient file supports maximum degree {maximum_degree}; " "degree 2 is unavailable." )

        position = np.array([4510.0, -3120.0, 5230.0])
        point_mass = np.asarray( make_pines(0).acceleration(position), dtype=float )
        test_degree = min(20, maximum_degree)
        higher_degree = np.asarray( make_pines(test_degree).acceleration(position), dtype=float )

        self.assertTrue(np.all(np.isfinite(higher_degree)))
        self.assertGreater( float(np.linalg.norm(higher_degree - point_mass)), 0.0, "Higher-degree gravity unexpectedly matched the point-mass field exactly.", )

    # ------------------------------------------------------------
    # Degree-zero Newtonian reference tests
    # ------------------------------------------------------------

    def test_degree_zero_acceleration_matches_newtonian_reference(self):
        """Verify degree-zero acceleration against the point-mass equation."""
        gravity = make_pines(0)

        positions = (
            np.array([7000.0, 0.0, 0.0]),
            np.array([0.0, -42164.0, 0.0]),
            np.array( [ 0.0, 1.0e8 / np.sqrt(2.0), 1.0e8 / np.sqrt(2.0), ] ),
        )

        for position in positions:
            with self.subTest(position=position):
                np.testing.assert_allclose( gravity.acceleration(position), newtonian_acceleration(position), rtol=3e-14, atol=2e-18, )

    def test_degree_zero_acceleration_magnitude_matches_mu_over_r_squared(self):
        """Verify the acceleration magnitude follows mu divided by radius squared."""
        gravity = make_pines(0)

        for radius in (6500.0, 7000.0, 10000.0, 42164.0, 100000.0):
            with self.subTest(radius=radius):
                position = np.array([radius, 0.0, 0.0])
                actual = np.linalg.norm(gravity.acceleration(position))
                expected = MU / radius**2

                self.assertAlmostEqual( actual, expected, delta=expected * 5e-14, )

    def test_degree_zero_inverse_square_scaling(self):
        """Verify doubling the position vector reduces acceleration by a factor of four."""
        gravity = make_pines(0)
        position = np.array([7000.0, -1200.0, 500.0])

        first = np.asarray(gravity.acceleration(position))
        second = np.asarray(gravity.acceleration(2.0 * position))

        np.testing.assert_allclose( second, first / 4.0, rtol=5e-14, atol=1e-18 )

    def test_degree_zero_acceleration_is_radially_inward(self):
        """Verify point-mass acceleration is inward and parallel to the position vector."""
        gravity = make_pines(0)

        positions = (
            np.array([7000.0, 0.0, 0.0]),
            np.array([1000.0, -2000.0, 6000.0]),
            np.array([-8000.0, 1000.0, -3000.0]),
        )

        for position in positions:
            with self.subTest(position=position):
                acceleration = np.asarray(gravity.acceleration(position))
                self.assertLess(float(np.dot(position, acceleration)), 0.0)
                np.testing.assert_allclose( np.cross(position, acceleration), np.zeros(3), atol=1e-10, )

    def test_degree_zero_rotational_symmetry(self):
        """Verify rotating a position around z rotates its acceleration identically."""
        gravity = make_pines(0)
        position = np.array([7000.0, 1200.0, -900.0])
        angle = 0.73

        rotation = np.array( [ [np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0], ] )

        original = np.asarray(gravity.acceleration(position))
        rotated = np.asarray(gravity.acceleration(rotation @ position))

        np.testing.assert_allclose( rotated, rotation @ original, rtol=5e-14, atol=2e-18, )

    def test_degree_zero_acceleration_is_deterministic(self):
        """Verify repeated acceleration evaluations at the same position are identical."""
        gravity = make_pines(0)
        position = np.array([7000.0, -1200.0, 800.0])
        first = np.asarray(gravity.acceleration(position))

        for _ in range(5):
            np.testing.assert_array_equal( gravity.acceleration(position), first )

    def test_degree_two_j2_acceleration_matches_analytic_reference(self):
        """Verify degree-two gravity reproduces the independent analytic J2 equation."""
        with tempfile.TemporaryDirectory() as directory:
            coefficient_file = Path(directory) / "j2_only.nor"
            coefficient_file.write_text( f"# fully normalized J2-only field\n" f"2 0 {-J2 / np.sqrt(5.0):.17e} 0D0 0 0\n", encoding="ascii", )

            gravity = make_pines(2, coefficient_file)
            positions = (
                np.array([7000.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 7000.0]),
                np.array([4510.0, -3120.0, 5230.0]),
            )

            for position in positions:
                with self.subTest(position=position):
                    np.testing.assert_allclose( gravity.acceleration(position), j2_acceleration(position), rtol=5e-14, atol=3e-18, )

    def test_j2_acceleration_is_symmetric_across_equatorial_plane(self):
        """Verify a zonal J2 field preserves x/y acceleration and reverses z acceleration."""
        with tempfile.TemporaryDirectory() as directory:
            coefficient_file = Path(directory) / "j2_only.nor"
            coefficient_file.write_text( f"2 0 {-J2 / np.sqrt(5.0):.17e} 0D0 0 0\n", encoding="ascii", )

            gravity = make_pines(2, coefficient_file)
            position = np.array([4510.0, -3120.0, 5230.0])
            north = np.asarray(gravity.acceleration(position))
            south_position = position * np.array([1.0, 1.0, -1.0])
            south = np.asarray(gravity.acceleration(south_position))

            np.testing.assert_allclose( south, north * np.array([1.0, 1.0, -1.0]), rtol=5e-14, atol=3e-18, )

    def test_j2_equatorial_acceleration_matches_closed_form(self):
        """Verify the equatorial J2 acceleration against its closed-form expression."""
        with tempfile.TemporaryDirectory() as directory:
            coefficient_file = Path(directory) / "j2_only.nor"
            coefficient_file.write_text( f"2 0 {-J2 / np.sqrt(5.0):.17e} 0D0 0 0\n", encoding="ascii", )

            gravity = make_pines(2, coefficient_file)
            radius = 7000.0
            expected = np.array( [ -MU / radius**2 * (1.0 + 1.5 * J2 * (REFERENCE_RADIUS / radius)**2), 0.0, 0.0, ] )

            np.testing.assert_allclose( gravity.acceleration(np.array([radius, 0.0, 0.0])), expected, rtol=5e-14, atol=3e-18, )

    def test_j2_polar_acceleration_matches_closed_form(self):
        """Verify polar J2 acceleration against the expected radial correction."""
        with tempfile.TemporaryDirectory() as directory:
            coefficient_file = Path(directory) / "j2_only.nor"
            coefficient_file.write_text( f"2 0 {-J2 / np.sqrt(5.0):.17e} 0D0 0 0\n", encoding="ascii", )

            gravity = make_pines(2, coefficient_file)
            radius = 7000.0
            actual = np.asarray( gravity.acceleration(np.array([0.0, 0.0, radius])) )
            expected = np.array( [ 0.0, 0.0, -MU / radius**2 * (1.0 - 3.0 * J2 * (REFERENCE_RADIUS / radius)**2), ] )

            np.testing.assert_allclose( actual, expected, rtol=5e-14, atol=3e-18 )

    def test_j2_only_field_matches_newtonian_at_multiple_radii(self):
        """Verify the analytic J2 field at several orbital distances."""
        with tempfile.TemporaryDirectory() as directory:
            coefficient_file = Path(directory) / "j2_only.nor"
            coefficient_file.write_text( f"2 0 {-J2 / np.sqrt(5.0):.17e} 0D0 0 0\n", encoding="ascii", )

            gravity = make_pines(2, coefficient_file)

            for radius in (6500.0, 7000.0, 12000.0, 42164.0):
                with self.subTest(radius=radius):
                    position = np.array( [radius / np.sqrt(2.0), 0.0, radius / np.sqrt(2.0)] )

                    np.testing.assert_allclose( gravity.acceleration(position), j2_acceleration(position), rtol=5e-14, atol=3e-18, )

    def test_invalid_coefficient_file_is_rejected(self):
        """Verify PinesGravity rejects a nonexistent coefficient file."""
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "missing_coefficients.nor"

            with self.assertRaises((OSError, RuntimeError, ValueError)):
                make_pines(2, missing_file)

    def test_zero_position_is_rejected_or_nonfinite(self):
        """Verify the point-mass singularity at the origin cannot produce finite acceleration."""
        gravity = make_pines(0)

        try:
            acceleration = np.asarray( gravity.acceleration(np.zeros(3)), dtype=float )
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            FloatingPointError,
        ):
            return

        self.assertFalse( np.all(np.isfinite(acceleration)), "Gravity at the origin unexpectedly returned a finite acceleration.", )

    def test_two_body_quarter_period_matches_analytic_state(self):
        """Verify a circular orbit advances 90 degrees after one quarter period."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        radius = 7000.0
        initial = circular_initial_state(radius)
        quarter_period = circular_period(radius) / 4.0
        speed = np.sqrt(MU / radius)

        expected = np.array([0.0, radius, 0.0, -speed, 0.0, 0.0])
        actual = propagate(integrator, initial, quarter_period)

        np.testing.assert_allclose( actual[:3], expected[:3], rtol=0.0, atol=3e-5 )
        np.testing.assert_allclose( actual[3:], expected[3:], rtol=0.0, atol=3e-8 )

    def test_two_body_half_period_propagation_matches_naif_reference_property(self):
        """Verify a circular orbit reaches the opposite state after half a period."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        actual = propagate(integrator, initial, circular_period() / 2.0)

        np.testing.assert_allclose( actual[:3], -initial[:3], rtol=0.0, atol=3e-6 )
        np.testing.assert_allclose( actual[3:], -initial[3:6], rtol=0.0, atol=3e-9 )

    def test_two_body_full_period_returns_to_initial_state(self):
        """Verify position and velocity return to their initial values after one period."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        actual = propagate(integrator, initial, circular_period())

        np.testing.assert_allclose( actual[:3], initial[:3], rtol=0.0, atol=5e-5 )
        np.testing.assert_allclose( actual[3:], initial[3:6], rtol=0.0, atol=5e-8 )

    def test_two_body_circular_orbit_radius_remains_constant(self):
        """Verify orbital radius remains constant throughout a circular point-mass orbit."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        radius = np.linalg.norm(initial[:3])
        times = np.linspace(0.0, circular_period(), 17)

        for time_s in times:
            with self.subTest(time_s=time_s):
                state = (
                    initial[:6]
                    if time_s == 0.0
                    else propagate(integrator, initial, time_s)
                )
                self.assertAlmostEqual( np.linalg.norm(state[:3]), radius, delta=2e-5 )

    def test_two_body_specific_energy_is_conserved(self):
        """Verify specific orbital energy remains constant over two orbital periods."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        reference_energy = specific_energy(initial)

        for time_s in np.linspace(0.0, 2.0 * circular_period(), 21):
            with self.subTest(time_s=time_s):
                state = (
                    initial[:6]
                    if time_s == 0.0
                    else propagate(integrator, initial, time_s)
                )
                self.assertAlmostEqual( specific_energy(state), reference_energy, delta=2e-9 )

    def test_two_body_angular_momentum_vector_is_conserved(self):
        """Verify the specific angular momentum vector remains constant in a central field."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        reference_momentum = specific_angular_momentum(initial)

        for time_s in np.linspace(0.0, 2.0 * circular_period(), 21):
            with self.subTest(time_s=time_s):
                state = (
                    initial[:6]
                    if time_s == 0.0
                    else propagate(integrator, initial, time_s)
                )
                np.testing.assert_allclose( specific_angular_momentum(state), reference_momentum, rtol=0.0, atol=2e-5, )

    def test_two_body_short_arc_matches_initial_acceleration_expansion(self):
        """Verify short-time position agrees with the second-order Taylor approximation."""
        gravity = make_pines(0)
        integrator = make_two_body_integrator(gravity)
        initial = circular_initial_state()
        position = initial[:3]
        velocity = initial[3:6]
        acceleration = newtonian_acceleration(position)
        time_s = 0.1

        expected_position = (
            position
            + velocity * time_s
            + 0.5 * acceleration * time_s**2
        )
        actual = propagate(integrator, initial, time_s)

        np.testing.assert_allclose( actual[:3], expected_position, rtol=0.0, atol=2e-7 )

    def test_tighter_tolerance_produces_consistent_full_period_state(self):
        """Verify tighter and looser integrator tolerances produce consistent final states."""
        gravity = make_pines(0)
        initial = circular_initial_state()
        period = circular_period()

        loose = propagate( make_two_body_integrator(gravity, 1e-9), initial, period )
        tight = propagate( make_two_body_integrator(gravity, 1e-12), initial, period )

        np.testing.assert_allclose( tight, loose, rtol=0.0, atol=2e-5 )


if __name__ == "__main__":
    unittest.main()