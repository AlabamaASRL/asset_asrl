# -*- coding: utf-8 -*-
"""
ASSET PINES GRAVITY REGRESSION AND GRAIL SPICE INTEGRATION TESTS

Regression tests for ASSET PinesGravity using GRAIL lunar gravity
models and real GRAIL SPICE ephemerides.
"""

import atexit
import csv
import html
import logging
import math
import os
import re
import tempfile
import unittest
import urllib.error
import urllib.request

from pathlib import Path
from urllib.parse import urljoin

import numpy as np

try:
    import asset_asrl as asset
except ImportError:
    import asset

import asset_asrl.Astro as astro

try:
    import spiceypy as spice
except ImportError as error:
    raise ImportError("spiceypy is required for the GRAIL SPICE integration tests.") from error

DOWNLOAD_TIMEOUT_SECONDS = 120
USER_AGENT = "ASSET-PinesGravity-UnitTests/1.0"
MINIMUM_VALID_FILE_SIZE = 100

AUTOMATIC_DOWNLOAD = os.environ.get("ASSET_GRAIL_DOWNLOAD", "1").lower() not in ("0", "false", "no")

LOG_LEVEL = os.environ.get("ASSET_GRAIL_LOG_LEVEL", "WARNING").upper()

LOGGER = logging.getLogger("ASSET_GRAIL_TESTS")
LOGGER.setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))

if not logging.getLogger().handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING), format="%(levelname)s | %(message)s")

# All downloaded data is stored outside the repository.
# The temporary directory is deleted automatically when Python exits.
TEMPORARY_DATA_DIRECTORY = Path(tempfile.mkdtemp(prefix="asset_grail_tests_"))
atexit.register(lambda: __import__("shutil").rmtree(TEMPORARY_DATA_DIRECTORY, ignore_errors=True))

COEFFICIENT_DIRECTORY = TEMPORARY_DATA_DIRECTORY / "coefficients"
SPICE_DIRECTORY = TEMPORARY_DATA_DIRECTORY / "spice"

COEFFICIENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
SPICE_DIRECTORY.mkdir(parents=True, exist_ok=True)

GRAIL_ARCHIVE_URL = "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/"

GRAIL_MODELS = {
    "GL0420A": {"filename": "jggrx_0420a_sha.tab", "maximum_degree": 420, "institution": "JPL"},
    "GL0660B": {"filename": "jggrx_0660b_sha.tab", "maximum_degree": 660, "institution": "JPL"},
    "GL0900C": {"filename": "jggrx_0900c_sha.tab", "maximum_degree": 900, "institution": "JPL"},
    "GL0900D": {"filename": "jggrx_0900d_sha.tab", "maximum_degree": 900, "institution": "JPL"},
    "GL1500E": {"filename": "jggrx_1500e_sha.tab", "maximum_degree": 1500, "institution": "JPL"},
    "GRGM660PRIM": {"filename": "gggrx_0660pm_sha.tab", "maximum_degree": 660, "institution": "GSFC"},
    "GRGM900C": {"filename": "gggrx_0900c_sha.tab", "maximum_degree": 900, "institution": "GSFC"},
    "GRGM1200A": {"filename": "gggrx_1200a_sha.tab", "maximum_degree": 1200, "institution": "GSFC"},
}

DEFAULT_TEST_MODELS = ("GL0420A", "GL0660B", "GRGM660PRIM", "GRGM1200A")

COMMON_TEST_DEGREE = 20
MODEL_COMPARISON_DEGREE = 20
HIGH_DEGREE_TEST = 180

ACCELERATION_POSITIONS = (
    (1838.0, 0.0, 0.0),
    (0.0, 1838.0, 0.0),
    (0.0, 0.0, 1838.0),
    (2000.0, 1000.0, 500.0),
    (2500.0, -1000.0, 750.0),
    (-2200.0, 1300.0, 900.0),
)

CIRCULAR_ORBIT_ALTITUDE = 500.0
CIRCULAR_ORBIT_STEP_SIZE = 2.0

GRAIL_SPICE_START_UTC = "2012-04-01T00:00:00"
GRAIL_SPICE_STOP_UTC = "2012-04-01T06:00:00"

GRAIL_SPICE_SAMPLE_INTERVAL_SECONDS = 60.0
GRAIL_SPICE_VELOCITY_DIFFERENCE_STEP_SECONDS = 1.0

GRAIL_SPICE_GRAVITY_DEGREE = int(os.environ.get("ASSET_GRAIL_SPICE_DEGREE", "360"))
GRAIL_SPICE_SAMPLE_COUNT = int(os.environ.get("ASSET_GRAIL_SPICE_SAMPLES", "10"))

GRAIL_SPICE_INERTIAL_FRAME = "J2000"
GRAIL_SPICE_GRAVITY_FRAME = "MOON_PA"
GRAIL_SPICE_ABERRATION_CORRECTION = "NONE"
GRAIL_SPICE_OBSERVER = "MOON"

GRAIL_SPICE_MINIMUM_RADIUS_KM = 1500.0
GRAIL_SPICE_MAXIMUM_RADIUS_KM = 3000.0

GRAIL_SPICE_ARCHIVE_ROOT = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/"
GENERIC_KERNEL_ROOT = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"

GRAIL_SPICE_KERNELS = {
    "grail_spk": SPICE_DIRECTORY / "grail_120301_120529_sci_v02.bsp",
    "de421": SPICE_DIRECTORY / "de421.bsp",
    "lsk": SPICE_DIRECTORY / "naif0012.tls",
    "pck": SPICE_DIRECTORY / "moon_pa_de421_1900-2050.bpc",
    "fk": SPICE_DIRECTORY / "moon_080317.tf",
}

GRAIL_SPICE_KERNEL_URLS = {
    "grail_spk": GRAIL_SPICE_ARCHIVE_ROOT + "data/spk/",
    "de421": GENERIC_KERNEL_ROOT + "spk/planets/a_old_versions/",
    "lsk": GENERIC_KERNEL_ROOT + "lsk/",
    "pck": GENERIC_KERNEL_ROOT + "pck/",
    "fk": GENERIC_KERNEL_ROOT + "fk/satellites/",
}


def get_selected_models():
    selection = os.environ.get("ASSET_GRAIL_TEST_MODELS", "").strip()

    if not selection:
        return list(DEFAULT_TEST_MODELS)

    if selection.lower() == "all":
        return list(GRAIL_MODELS)

    selected = [name.strip().upper() for name in selection.split(",")]
    invalid = [name for name in selected if name not in GRAIL_MODELS]

    if invalid:
        raise ValueError(f"Unknown GRAIL model selection: {invalid}")

    return selected


SELECTED_MODELS = get_selected_models()

def download_url(url, destination, minimum_size=MINIMUM_VALID_FILE_SIZE):
    """Download a URL to a temporary cache file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size >= minimum_size:
        return destination

    if not AUTOMATIC_DOWNLOAD:
        raise FileNotFoundError(f"Required file is not cached and automatic downloads are disabled: {destination}")

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    LOGGER.info("Downloading %s", url)

    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

        if not temporary_path.is_file() or temporary_path.stat().st_size < minimum_size:
            raise RuntimeError(f"Downloaded file is too small or empty: {temporary_path}")

        temporary_path.replace(destination)
        return destination

    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to download {url}: {error}") from error


def download_directory_file(directory_url, filename_pattern, destination, minimum_size=MINIMUM_VALID_FILE_SIZE):
    """Find a matching filename in an HTTP directory listing and download it."""

    destination = Path(destination)

    if destination.is_file() and destination.stat().st_size >= minimum_size:
        return destination

    if not AUTOMATIC_DOWNLOAD:
        raise FileNotFoundError(f"Required SPICE kernel is not cached: {destination}")

    request = urllib.request.Request(directory_url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            page = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Unable to inspect kernel directory {directory_url}: {error}") from error

    links = re.findall(r'href=["\']([^"\']+)["\']', page, flags=re.IGNORECASE)
    filenames = [html.unescape(link.split("/")[-1]) for link in links]
    matches = [name for name in filenames if re.fullmatch(filename_pattern, name, flags=re.IGNORECASE)]

    if not matches:
        raise FileNotFoundError(f"No file matching {filename_pattern} was found in {directory_url}")

    selected_filename = sorted(matches)[-1]
    file_url = urljoin(directory_url, selected_filename)

    return download_url(file_url, destination, minimum_size)


def is_valid_coefficient_file(filename):
    path = Path(filename)

    if not path.is_file() or path.stat().st_size < MINIMUM_VALID_FILE_SIZE:
        return False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            sample = [stream.readline() for _ in range(30)]

        content = "".join(sample).lower()

        return bool(any(line.strip() for line in sample)) and "<html" not in content and "<!doctype html" not in content

    except OSError:
        return False


def download_coefficient_file(model_name):
    if model_name not in GRAIL_MODELS:
        raise ValueError(f"Unknown GRAIL model: {model_name}")

    filename = GRAIL_MODELS[model_name]["filename"]
    destination = COEFFICIENT_DIRECTORY / filename

    if is_valid_coefficient_file(destination):
        LOGGER.debug("Using cached coefficient file: %s", destination)
        return destination

    if destination.exists():
        destination.unlink()

    download_url(GRAIL_ARCHIVE_URL + filename, destination)

    if not is_valid_coefficient_file(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded coefficient file is invalid: {filename}")

    return destination


def parse_grail_metadata(filename):
    with Path(filename).open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()

            if not stripped or stripped.lower() == "grail" or stripped.startswith(("#", "%", "!")):
                continue

            fields = next(csv.reader([stripped], skipinitialspace=True))

            if len(fields) < 8:
                continue

            try:
                values = [float(value.strip().replace("D", "E").replace("d", "e")) for value in fields[:8]]
            except ValueError:
                continue

            radius, mu, uncertainty, degree, order, norm_code, longitude, latitude = values

            degree = int(degree)
            order = int(order)
            norm_code = int(norm_code)

            if radius <= 0.0 or mu <= 0.0 or degree < 2 or order < 0:
                continue

            normalization = {0: "unnormalized", 1: "fully_normalized"}.get(norm_code, "unknown")

            return {
                "reference_radius": radius,
                "mu": mu,
                "mu_uncertainty": uncertainty,
                "maximum_degree": degree,
                "maximum_order": order,
                "normalization_code": norm_code,
                "normalization": normalization,
                "reference_longitude": longitude,
                "reference_latitude": latitude,
                "source_file": str(filename),
            }

    raise ValueError(f"Could not parse GRAIL metadata from {filename}")


def make_pines(model_name, degree=COMMON_TEST_DEGREE):
    coefficient_file = download_coefficient_file(model_name)
    metadata = parse_grail_metadata(coefficient_file)

    if metadata["normalization"] == "unknown":
        raise ValueError(f"Unsupported normalization code in {model_name}")

    maximum_supported_degree = min(metadata["maximum_degree"], metadata["maximum_order"])

    if degree < 0 or degree > maximum_supported_degree:
        raise ValueError(f"{model_name} does not support degree/order {degree}")

    gravity = astro.PinesGravity(str(coefficient_file), int(degree), float(metadata["mu"]), float(metadata["reference_radius"]), metadata["normalization"])

    return gravity, metadata


def evaluate_acceleration(gravity, position):
    position = np.asarray(position, dtype=float)

    if position.shape != (3,):
        raise ValueError(f"Expected a three-element position vector, received {position.shape}")

    return np.asarray(gravity.acceleration(position), dtype=float).reshape(3)


def newtonian_acceleration(position, mu):
    position = np.asarray(position, dtype=float)
    radius = np.linalg.norm(position)

    if radius == 0.0:
        raise ValueError("Newtonian acceleration is singular at the origin.")

    return -float(mu) * position / radius**3

def specific_orbital_energy(position, velocity, mu):
    return 0.5 * np.dot(velocity, velocity) - float(mu) / np.linalg.norm(position)

def specific_angular_momentum(position, velocity):
    return np.cross(position, velocity)

def download_grail_spice_kernels():
    """Download the GRAIL SPK and required generic SPICE kernels."""

    download_directory_file(GRAIL_SPICE_KERNEL_URLS["grail_spk"], r"grail_120301_120529_sci_v02\.bsp", GRAIL_SPICE_KERNELS["grail_spk"])
    download_directory_file(GRAIL_SPICE_KERNEL_URLS["de421"], r"de421\.bsp", GRAIL_SPICE_KERNELS["de421"])
    download_directory_file(GRAIL_SPICE_KERNEL_URLS["lsk"], r"naif0012\.tls", GRAIL_SPICE_KERNELS["lsk"])
    download_directory_file(GRAIL_SPICE_KERNEL_URLS["pck"], r"moon_pa_de421_1900-2050\.bpc", GRAIL_SPICE_KERNELS["pck"])
    download_directory_file(GRAIL_SPICE_KERNEL_URLS["fk"], r"moon_080317\.tf", GRAIL_SPICE_KERNELS["fk"])

    for name, path in GRAIL_SPICE_KERNELS.items():
        if not path.is_file() or path.stat().st_size < MINIMUM_VALID_FILE_SIZE:
            raise FileNotFoundError(f"Missing or invalid SPICE kernel {name}: {path}")

        LOGGER.debug("SPICE kernel ready: %s", path)

    return GRAIL_SPICE_KERNELS

def load_grail_spice_kernels():
    """Load kernels in the required dependency order."""

    kernels = download_grail_spice_kernels()
    spice.kclear()

    for name in ("lsk", "de421", "pck", "fk", "grail_spk"):
        spice.furnsh(str(kernels[name]))

    return kernels

def get_grail_spacecraft_ids(spk_filename):
    """Return NAIF object IDs contained in the GRAIL SPK."""
    return sorted(int(object_id) for object_id in spice.spkobj(str(spk_filename)))

def get_spacecraft_state(spacecraft_id, epoch):
    """Return a spacecraft state relative to the Moon in J2000."""
    state, light_time = spice.spkezr(str(spacecraft_id), float(epoch), GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_ABERRATION_CORRECTION, GRAIL_SPICE_OBSERVER)
    return np.asarray(state, dtype=float), float(light_time)

def get_spacecraft_acceleration(spacecraft_id, epoch, step_size=GRAIL_SPICE_VELOCITY_DIFFERENCE_STEP_SECONDS):
    """Estimate inertial acceleration using centered finite differences."""
    state_before, _ = get_spacecraft_state(spacecraft_id, epoch - step_size)
    state_after, _ = get_spacecraft_state(spacecraft_id, epoch + step_size)
    return (state_after[3:] - state_before[3:]) / (2.0 * step_size)


def transform_state(state, source_frame, destination_frame, epoch):
    """Transform a six-dimensional state using a SPICE state transformation."""
    state = np.asarray(state, dtype=float).reshape(6)
    transformation = np.asarray(spice.sxform(source_frame, destination_frame, float(epoch)), dtype=float)
    return transformation @ state

def transform_vector(vector, source_frame, destination_frame, epoch):
    """Transform a physical three-dimensional vector between frames."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    transformation = np.asarray(spice.pxform(source_frame, destination_frame, float(epoch)), dtype=float)
    return transformation @ vector


def get_spice_sample_epochs(start_utc=GRAIL_SPICE_START_UTC, stop_utc=GRAIL_SPICE_STOP_UTC, interval=GRAIL_SPICE_SAMPLE_INTERVAL_SECONDS):
    """Generate ET sample epochs over the requested UTC interval."""
    start_et = float(spice.str2et(start_utc))
    stop_et = float(spice.str2et(stop_utc))

    if stop_et <= start_et:
        raise ValueError("The SPICE stop epoch must be after the start epoch.")

    epochs = np.arange(start_et, stop_et + 0.5 * interval, interval, dtype=float)

    if epochs.size == 0:
        raise RuntimeError("No SPICE epochs were generated.")

    return epochs


def find_grail_spacecraft(spacecraft_ids, epochs):
    """Find spacecraft with valid SPK states in the lunar radius interval."""
    candidates = []

    for spacecraft_id in spacecraft_ids:
        valid_samples = []

        for epoch in epochs:
            try:
                state, _ = get_spacecraft_state(spacecraft_id, epoch)
                radius = float(np.linalg.norm(state[:3]))

                if GRAIL_SPICE_MINIMUM_RADIUS_KM <= radius <= GRAIL_SPICE_MAXIMUM_RADIUS_KM:
                    valid_samples.append((float(epoch), state, radius))

            except Exception as error:
                LOGGER.debug("SPK state unavailable for ID %s at ET %.3f: %s", spacecraft_id, epoch, error)

        if valid_samples:
            candidates.append({"spacecraft_id": spacecraft_id, "samples": valid_samples})

    return candidates


class GRAILMetadataTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_data = {}

        for name in SELECTED_MODELS:
            path = download_coefficient_file(name)
            metadata = parse_grail_metadata(path)
            cls.model_data[name] = {"file": path, "metadata": metadata}

    def test_selected_files_exist(self):
        for name, data in self.model_data.items():
            with self.subTest(model=name):
                self.assertTrue(data["file"].is_file())
                self.assertGreater(data["file"].stat().st_size, MINIMUM_VALID_FILE_SIZE)

    def test_metadata_constants_and_lunar_ranges(self):
        for name, data in self.model_data.items():
            metadata = data["metadata"]

            with self.subTest(model=name):
                self.assertGreater(metadata["mu"], 4800.0)
                self.assertLess(metadata["mu"], 5000.0)
                self.assertGreater(metadata["reference_radius"], 1700.0)
                self.assertLess(metadata["reference_radius"], 1800.0)

    def test_metadata_degree_order_and_catalog(self):
        for name, data in self.model_data.items():
            metadata = data["metadata"]

            with self.subTest(model=name):
                self.assertEqual(metadata["maximum_degree"], GRAIL_MODELS[name]["maximum_degree"])
                self.assertGreaterEqual(metadata["maximum_order"], 0)
                self.assertLessEqual(metadata["maximum_order"], metadata["maximum_degree"])

    def test_metadata_normalization_is_supported(self):
        for name, data in self.model_data.items():
            with self.subTest(model=name):
                self.assertIn(data["metadata"]["normalization"], ("fully_normalized", "unnormalized"))


class GRAILPinesGravityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = {}

        for name in SELECTED_MODELS:
            gravity, metadata = make_pines(name)
            cls.models[name] = {"gravity": gravity, "metadata": metadata}

    def test_all_models_construct(self):
        for name, data in self.models.items():
            with self.subTest(model=name):
                self.assertIsNotNone(data["gravity"])

    def test_accelerations_are_finite(self):
        for name, data in self.models.items():
            for position in ACCELERATION_POSITIONS:
                with self.subTest(model=name, position=position):
                    acceleration = evaluate_acceleration(data["gravity"], position)
                    self.assertEqual(acceleration.shape, (3,))
                    self.assertTrue(np.all(np.isfinite(acceleration)))

    def test_accelerations_are_deterministic(self):
        for name, data in self.models.items():
            for position in ACCELERATION_POSITIONS:
                with self.subTest(model=name, position=position):
                    first = evaluate_acceleration(data["gravity"], position)
                    second = evaluate_acceleration(data["gravity"], position)
                    np.testing.assert_allclose(first, second, rtol=0.0, atol=1.0e-13)

    def test_degree_zero_matches_newtonian_gravity(self):
        for name, data in self.models.items():
            gravity, _ = make_pines(name, 0)

            for position in ACCELERATION_POSITIONS:
                with self.subTest(model=name, position=position):
                    actual = evaluate_acceleration(gravity, position)
                    expected = newtonian_acceleration(position, data["metadata"]["mu"])
                    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-12)

    def test_inverse_square_scaling(self):
        for name in self.models:
            gravity, _ = make_pines(name, 0)
            position = np.array([2000.0, 0.0, 0.0])
            first = evaluate_acceleration(gravity, position)
            second = evaluate_acceleration(gravity, 2.0 * position)
            np.testing.assert_allclose(second, first / 4.0, rtol=1.0e-10, atol=1.0e-12)

    def test_degree_zero_acceleration_is_radial(self):
        for name in self.models:
            gravity, _ = make_pines(name, 0)

            for position in ACCELERATION_POSITIONS:
                acceleration = evaluate_acceleration(gravity, position)
                error = np.linalg.norm(np.cross(position, acceleration))

                with self.subTest(model=name, position=position):
                    self.assertLess(error, 1.0e-8)

    def test_degree_zero_rotational_symmetry(self):
        for name in self.models:
            gravity, _ = make_pines(name, 0)
            first = evaluate_acceleration(gravity, [2000.0, 0.0, 0.0])
            second = evaluate_acceleration(gravity, [0.0, 2000.0, 0.0])
            expected = np.array([0.0, first[0], 0.0])
            np.testing.assert_allclose(second, expected, rtol=1.0e-10, atol=1.0e-12)

    def test_higher_degree_field_differs_from_point_mass(self):
        for name in self.models:
            high, _ = make_pines(name, COMMON_TEST_DEGREE)
            point, _ = make_pines(name, 0)
            position = np.array([1838.0, 100.0, 50.0])
            difference = np.linalg.norm(evaluate_acceleration(high, position) - evaluate_acceleration(point, position))
            self.assertGreater(difference, 0.0)

    def test_high_degree_accelerations_are_finite(self):
        for name in self.models:
            degree = min(HIGH_DEGREE_TEST, GRAIL_MODELS[name]["maximum_degree"])
            gravity, _ = make_pines(name, degree)

            for position in ACCELERATION_POSITIONS:
                with self.subTest(model=name, degree=degree, position=position):
                    acceleration = evaluate_acceleration(gravity, position)
                    self.assertTrue(np.all(np.isfinite(acceleration)))

    def test_origin_is_rejected(self):
        for name in self.models:
            gravity, _ = make_pines(name)

            with self.subTest(model=name):
                with self.assertRaises((ValueError, RuntimeError, ZeroDivisionError)):
                    evaluate_acceleration(gravity, (0.0, 0.0, 0.0))


class GRAILModelComparisonTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models = {name: make_pines(name, MODEL_COMPARISON_DEGREE)[0] for name in SELECTED_MODELS}

    def test_model_accelerations_and_differences_are_finite(self):
        if len(self.models) < 2:
            self.skipTest("At least two GRAIL models are required for comparison.")

        for name, gravity in self.models.items():
            for position in ACCELERATION_POSITIONS:
                with self.subTest(model=name, position=position):
                    acceleration = evaluate_acceleration(gravity, position)
                    self.assertTrue(np.all(np.isfinite(acceleration)))

        names = list(self.models)

        for i, first_name in enumerate(names):
            for second_name in names[i + 1:]:
                for position in ACCELERATION_POSITIONS:
                    first = evaluate_acceleration(self.models[first_name], position)
                    second = evaluate_acceleration(self.models[second_name], position)
                    difference = np.linalg.norm(first - second)
                    relative = difference / max(np.linalg.norm(first), np.linalg.norm(second))

                    with self.subTest(first_model=first_name, second_model=second_name, position=position):
                        self.assertTrue(np.isfinite(difference))
                        self.assertTrue(np.isfinite(relative))
                        self.assertGreaterEqual(difference, 0.0)


class GRAILCircularOrbitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gravity, cls.metadata = make_pines("GL0660B", 20)
        cls.central_gravity, _ = make_pines("GL0660B", 0)
        cls.mu = float(cls.metadata["mu"])
        cls.reference_radius = float(cls.metadata["reference_radius"])
        cls.orbit_radius = cls.reference_radius + CIRCULAR_ORBIT_ALTITUDE
        cls.circular_speed = math.sqrt(cls.mu / cls.orbit_radius)
        cls.circular_period = 2.0 * math.pi * math.sqrt(cls.orbit_radius**3 / cls.mu)

    def make_integrator(self, gravity, tolerance):
        """Construct an ASSET integrator for a six-state translational ODE."""
        args = asset.VectorFunctions.Arguments(7)
        rhs = asset.VectorFunctions.stack(args.segment3(3), gravity(args.head3()))
        ode = asset.OptimalControl.ode_6.ode(rhs, 6)
        integrator = ode.integrator(CIRCULAR_ORBIT_STEP_SIZE)
        integrator.setAbsTol(float(tolerance))
        return integrator

    def propagate(self, duration, tolerance=1.0e-10, gravity=None):
        """Propagate [x,y,z,vx,vy,vz,t] to the requested absolute final time."""
        duration = float(duration)

        if duration <= 0.0:
            raise ValueError("Propagation duration must be positive.")

        initial_state = np.array([self.orbit_radius, 0.0, 0.0, 0.0, self.circular_speed, 0.0])
        state = np.append(initial_state, 0.0).reshape(7, 1)
        selected_gravity = gravity if gravity is not None else self.gravity
        integrator = self.make_integrator(selected_gravity, tolerance)
        final_times = list(np.arange(CIRCULAR_ORBIT_STEP_SIZE, duration, CIRCULAR_ORBIT_STEP_SIZE))

        if not final_times or not math.isclose(final_times[-1], duration, rel_tol=0.0, abs_tol=1.0e-12):
            final_times.append(duration)

        for final_time in final_times:
            state = np.asarray(integrator.integrate(state, float(final_time)), dtype=float).reshape(7, 1)

        return initial_state, state[:6, 0].copy()

    def test_circular_orbit_initial_state_is_finite(self):
        initial = np.array([self.orbit_radius, 0.0, 0.0, 0.0, self.circular_speed, 0.0])
        self.assertTrue(np.all(np.isfinite(initial)))
        self.assertGreater(self.orbit_radius, self.reference_radius)

    def test_short_arc_propagation_is_finite(self):
        _, final = self.propagate(60.0)
        self.assertEqual(final.shape, (6,))
        self.assertTrue(np.all(np.isfinite(final)))

    def test_short_arc_radius_remains_reasonable(self):
        _, final = self.propagate(60.0)
        error = abs(np.linalg.norm(final[:3]) - self.orbit_radius)
        self.assertLess(error, 1.0)

    def test_specific_energy_is_conserved_in_central_field(self):
        initial, final = self.propagate(60.0, gravity=self.central_gravity)
        first = specific_orbital_energy(initial[:3], initial[3:], self.mu)
        last = specific_orbital_energy(final[:3], final[3:], self.mu)
        error = abs(last - first) / abs(first)
        self.assertLess(error, 1.0e-5)

    def test_specific_angular_momentum_is_conserved_in_central_field(self):
        initial, final = self.propagate(60.0, gravity=self.central_gravity)
        first = specific_angular_momentum(initial[:3], initial[3:])
        last = specific_angular_momentum(final[:3], final[3:])
        error = np.linalg.norm(last - first) / np.linalg.norm(first)
        self.assertLess(error, 1.0e-5)

    def test_tighter_integrator_tolerance_produces_finite_state(self):
        _, loose = self.propagate(60.0, 1.0e-9)
        _, tight = self.propagate(60.0, 1.0e-12)
        difference = np.linalg.norm(loose - tight)
        self.assertTrue(np.all(np.isfinite(loose)))
        self.assertTrue(np.all(np.isfinite(tight)))
        self.assertTrue(np.isfinite(difference))


class GRAILSPICEIntegrationTests(unittest.TestCase):
    """Integration tests using the real GRAIL SPK and lunar SPICE kernels."""

    @classmethod
    def setUpClass(cls):
        cls.kernels = load_grail_spice_kernels()
        cls.epochs = get_spice_sample_epochs()
        cls.start_et = float(cls.epochs[0])
        cls.stop_et = float(cls.epochs[-1])
        cls.spacecraft_ids = get_grail_spacecraft_ids(cls.kernels["grail_spk"])

        if not cls.spacecraft_ids:
            raise RuntimeError("No spacecraft object IDs were found in the GRAIL SPK.")

        cls.spacecraft_candidates = find_grail_spacecraft(cls.spacecraft_ids, cls.epochs)

        if not cls.spacecraft_candidates:
            raise RuntimeError("No GRAIL spacecraft states were found within the configured lunar radius interval.")

        cls.spacecraft_id = cls.spacecraft_candidates[0]["spacecraft_id"]
        cls.samples = cls.spacecraft_candidates[0]["samples"]

        if GRAIL_SPICE_SAMPLE_COUNT > 0:
            cls.samples = cls.samples[:GRAIL_SPICE_SAMPLE_COUNT]

        coefficient_model = os.environ.get("ASSET_GRAIL_SPICE_MODEL", "GL0420A").upper()

        if coefficient_model not in GRAIL_MODELS:
            raise ValueError(f"Unknown SPICE comparison model: {coefficient_model}")

        cls.gravity_model_name = coefficient_model
        spice_degree = min(GRAIL_SPICE_GRAVITY_DEGREE, GRAIL_MODELS[coefficient_model]["maximum_degree"])
        cls.gravity, cls.metadata = make_pines(coefficient_model, spice_degree)

        LOGGER.info("SPICE integration initialized: spacecraft=%s, model=%s, degree=%s, samples=%s", cls.spacecraft_id, cls.gravity_model_name, spice_degree, len(cls.samples))

    @classmethod
    def tearDownClass(cls):
        try:
            spice.kclear()
        except Exception:
            LOGGER.debug("Unable to clear SPICE kernels during teardown.", exc_info=True)

    def test_spice_kernels_are_loaded(self):
        self.assertGreater(spice.ktotal("ALL"), 0)
        self.assertTrue(self.kernels["grail_spk"].is_file())
        self.assertTrue(self.kernels["de421"].is_file())

    def test_grail_spacecraft_ids_are_available(self):
        self.assertGreater(len(self.spacecraft_ids), 0)
        self.assertIn(self.spacecraft_id, self.spacecraft_ids)

    def test_spk_sample_states_are_finite(self):
        self.assertGreater(len(self.samples), 0)

        for epoch, state, radius in self.samples:
            with self.subTest(epoch=epoch):
                self.assertEqual(state.shape, (6,))
                self.assertTrue(np.all(np.isfinite(state)))
                self.assertGreater(radius, GRAIL_SPICE_MINIMUM_RADIUS_KM)
                self.assertLess(radius, GRAIL_SPICE_MAXIMUM_RADIUS_KM)

    def test_spice_frame_transform_is_finite_and_orthogonal(self):
        epoch, state, radius = self.samples[0]
        rotation = np.asarray(spice.pxform(GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch), dtype=float)
        state_transform = np.asarray(spice.sxform(GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch), dtype=float)

        self.assertEqual(rotation.shape, (3, 3))
        self.assertEqual(state_transform.shape, (6, 6))
        self.assertTrue(np.all(np.isfinite(rotation)))
        self.assertTrue(np.all(np.isfinite(state_transform)))
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), rtol=0.0, atol=1.0e-10)

    def test_state_transform_round_trip(self):
        epoch, state, radius = self.samples[0]
        transformed = transform_state(state, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
        recovered = transform_state(transformed, GRAIL_SPICE_GRAVITY_FRAME, GRAIL_SPICE_INERTIAL_FRAME, epoch)
        error = np.linalg.norm(recovered - state)

        self.assertTrue(np.all(np.isfinite(transformed)))
        self.assertTrue(np.all(np.isfinite(recovered)))
        self.assertLess(error, 1.0e-7)

    def test_gravity_frame_position_is_finite(self):
        for epoch, state, radius in self.samples:
            with self.subTest(epoch=epoch):
                gravity_state = transform_state(state, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
                self.assertEqual(gravity_state.shape, (6,))
                self.assertTrue(np.all(np.isfinite(gravity_state)))
                self.assertGreater(np.linalg.norm(gravity_state[:3]), 0.0)

    def test_pines_gravity_returns_finite_acceleration_at_grail_positions(self):
        for epoch, state, radius in self.samples:
            gravity_state = transform_state(state, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
            acceleration = evaluate_acceleration(self.gravity, gravity_state[:3])

            with self.subTest(epoch=epoch):
                self.assertEqual(acceleration.shape, (3,))
                self.assertTrue(np.all(np.isfinite(acceleration)))
                self.assertGreater(np.linalg.norm(acceleration), 0.0)

    def test_spk_velocity_finite_difference_acceleration_is_finite(self):
        for epoch, state, radius in self.samples:
            acceleration = get_spacecraft_acceleration(self.spacecraft_id, epoch)

            with self.subTest(epoch=epoch):
                self.assertEqual(acceleration.shape, (3,))
                self.assertTrue(np.all(np.isfinite(acceleration)))
                self.assertGreater(np.linalg.norm(acceleration), 0.0)

    def test_acceleration_vector_frame_transform_preserves_norm(self):
        for epoch, state, radius in self.samples:
            inertial_acceleration = get_spacecraft_acceleration(self.spacecraft_id, epoch)
            gravity_frame_acceleration = transform_vector(inertial_acceleration, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
            difference = abs(np.linalg.norm(inertial_acceleration) - np.linalg.norm(gravity_frame_acceleration))

            with self.subTest(epoch=epoch):
                self.assertTrue(np.all(np.isfinite(gravity_frame_acceleration)))
                self.assertLess(difference, 1.0e-10)

    def test_pines_and_spk_acceleration_comparison_is_finite(self):
        """Compare acceleration magnitudes without assuming SPK is gravity-only."""
        relative_errors = []

        for epoch, state, radius in self.samples:
            gravity_state = transform_state(state, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
            pines_acceleration = evaluate_acceleration(self.gravity, gravity_state[:3])
            spk_acceleration_inertial = get_spacecraft_acceleration(self.spacecraft_id, epoch)
            spk_acceleration = transform_vector(spk_acceleration_inertial, GRAIL_SPICE_INERTIAL_FRAME, GRAIL_SPICE_GRAVITY_FRAME, epoch)
            pines_norm = float(np.linalg.norm(pines_acceleration))
            spk_norm = float(np.linalg.norm(spk_acceleration))
            absolute_difference = float(np.linalg.norm(pines_acceleration - spk_acceleration))
            relative_difference = absolute_difference / max(pines_norm, spk_norm, 1.0e-15)
            relative_errors.append(relative_difference)

            with self.subTest(epoch=epoch):
                self.assertTrue(np.all(np.isfinite(pines_acceleration)))
                self.assertTrue(np.all(np.isfinite(spk_acceleration)))
                self.assertTrue(np.isfinite(absolute_difference))
                self.assertTrue(np.isfinite(relative_difference))
                self.assertGreater(pines_norm, 0.0)
                self.assertGreater(spk_norm, 0.0)

        self.assertGreater(len(relative_errors), 0)
        LOGGER.info("PinesGravity/SPK acceleration comparison: mean relative difference=%.6e, maximum=%.6e", np.mean(relative_errors), np.max(relative_errors))

    def test_grail_epoch_coverage(self):
        """Confirm the requested sample epochs lie within the SPK coverage."""
        for epoch, state, radius in self.samples:
            with self.subTest(epoch=epoch):
                self.assertTrue(np.isfinite(epoch))
                self.assertTrue(np.all(np.isfinite(state)))
                self.assertGreaterEqual(epoch, self.start_et)
                self.assertLessEqual(epoch, self.stop_et)


class GRAILDownloadUtilityTests(unittest.TestCase):

    def test_invalid_model_name_is_rejected(self):
        with self.assertRaises(ValueError):
            download_coefficient_file("NOT_A_GRAIL_MODEL")

    def test_nonexistent_file_is_invalid(self):
        self.assertFalse(is_valid_coefficient_file(TEMPORARY_DATA_DIRECTORY / "file_that_does_not_exist.tab"))

    def test_empty_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "empty.tab"
            filename.write_text("", encoding="utf-8")
            self.assertFalse(is_valid_coefficient_file(filename))

    def test_html_file_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "invalid.tab"
            filename.write_text("<html><body>Not a coefficient file</body></html>", encoding="utf-8")
            self.assertFalse(is_valid_coefficient_file(filename))

def run_tests():
    """Run the complete test suite, including real GRAIL SPICE tests."""
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result

if __name__ == "__main__":
    test_result = run_tests()
    raise SystemExit(0 if test_result.wasSuccessful() else 1)