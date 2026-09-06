"""Unit tests for the Astro trajectory-file I/O helpers."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import types
import unittest

import numpy as np


# Resolve the production modules without importing asset_asrl's extension-backed root package.
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASTRO_ROOT = PACKAGE_ROOT / "Astro"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# DataReadWrite imports Constants through its package name.  Build the minimum
# package structure so these tests remain independent of the compiled extension.
asset_asrl = types.ModuleType("asset_asrl")
astro = types.ModuleType("asset_asrl.Astro")
asset_asrl.Astro = astro
sys.modules.setdefault("asset_asrl", asset_asrl)
sys.modules.setdefault("asset_asrl.Astro", astro)
constants = load_module("asset_asrl.Astro.Constants", ASTRO_ROOT / "Constants.py")
astro.Constants = constants
data_io = load_module("asset_asrl.Astro.DataReadWrite", ASTRO_ROOT / "DataReadWrite.py")


class DataReadWriteTests(unittest.TestCase):
    def test_write_and_read_data_round_trip_creates_target_folder(self):
        # Persisted trajectories must survive an on-disk round trip unchanged.
        trajectory = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_folder = pathlib.Path(temporary_directory) / "nested" / "trajectory_data"
            data_io.WriteData(trajectory, "sample", str(output_folder))

            self.assertTrue((output_folder / "sample.npy").is_file())
            np.testing.assert_array_equal(data_io.ReadData("sample", str(output_folder)), trajectory)

    def test_read_copernicus_file_scales_state_and_time_columns(self):
        # Copernicus exports km and days; ASSET uses metres and seconds.
        header = "header\n" * 5
        row = "unused,42.5,unused,1.25,1,2,3,4,5,6\n"
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_file = pathlib.Path(temporary_directory) / "copernicus.csv"
            input_file.write_text(header + row, encoding="utf-8")

            states = data_io.ReadCopernicusFile(input_file.name, str(input_file.parent))

        self.assertEqual(len(states), 1)
        np.testing.assert_array_equal(
            states[0],
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 1.25 * constants.day, 42.5],
        )

    def test_read_copernicus_file_ignores_the_five_line_header(self):
        # Metadata rows are not part of the spacecraft state history.
        header = "metadata\n" * 5
        rows = (
            "x,1,x,0,1,1,1,1,1,1\n"
            "x,2,x,2,2,2,2,2,2,2\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_file = pathlib.Path(temporary_directory) / "multiple_rows.csv"
            input_file.write_text(header + rows, encoding="utf-8")
            states = data_io.ReadCopernicusFile(input_file.name, str(input_file.parent))

        self.assertEqual(len(states), 2)
        self.assertEqual(states[0][6], 0.0)
        self.assertEqual(states[1][6], 2.0 * constants.day)
        self.assertEqual(states[1][7], 2.0)


if __name__ == "__main__":
    unittest.main()
