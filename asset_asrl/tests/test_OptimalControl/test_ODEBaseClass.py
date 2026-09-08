"""Unit tests for pure Python helpers in OptimalControl.ODEBaseClass."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = PACKAGE_ROOT / "OptimalControl" / "ODEBaseClass.py"

# The helpers below do not require the compiled ASSET extension.  Supply the
# minimal module name needed while importing this Python-only utility layer.
sys.modules.setdefault("asset", types.ModuleType("asset"))
spec = importlib.util.spec_from_file_location("asset_asrl_ode_base", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ODEBase = module.ODEBase


class FakeODE:
    # Minimal native-ODE stand-in; the helper methods only need named index groups.
    def __init__(self):
        self.groups = {"position": [0, 1, 2], "velocity": [3, 4, 5], "mass": [6]}
        self.added_groups = []

    def idx(self, name):
        return self.groups[name]

    def add_idx(self, name, idxs):
        self.added_groups.append((name, idxs))


def helper_with_fake_ode():
    # Bypass ODEBase.__init__, which constructs an extension-backed ODE object.
    helper = object.__new__(ODEBase)
    helper.ode = FakeODE()
    helper.XtUPVars = lambda: 7
    return helper


class ODEBaseHelperTests(unittest.TestCase):
    def test_make_index_set_flattens_nested_indices(self):
        # Groups may be assembled from nested scalar and sequence specifications.
        helper = helper_with_fake_ode()
        self.assertEqual(helper._make_index_set([0, [2, 4], np.int32(6)]), [0, 2, 4, 6])

    def test_make_index_set_rejects_empty_and_invalid_inputs(self):
        helper = helper_with_fake_ode()
        with self.assertRaisesRegex(Exception, "empty"):
            helper._make_index_set([])
        with self.assertRaisesRegex(Exception, "Invalid index"):
            helper._make_index_set("position")

    def test_add_vgroups_accepts_single_and_multiple_names(self):
        helper = helper_with_fake_ode()
        helper.add_Vgroups({"state": [0, 1], ("x", "y"): 2})
        self.assertEqual(
            helper.ode.added_groups,
            [("state", [0, 1]), ("x", [2]), ("y", [2])],
        )

    def test_make_units_broadcasts_scalars_and_uses_vectors(self):
        # A scalar unit applies to every component, while vectors preserve component scales.
        helper = helper_with_fake_ode()
        units = helper.make_units(position=1000.0, velocity=[10.0, 20.0, 30.0], mass=500.0)
        np.testing.assert_array_equal(units, [1000.0, 1000.0, 1000.0, 10.0, 20.0, 30.0, 500.0])

    def test_make_input_places_variable_groups_in_order(self):
        helper = helper_with_fake_ode()
        state = helper.make_input(position=[1.0, 2.0, 3.0], velocity=4.0, mass=500.0)
        np.testing.assert_array_equal(state, [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 500.0])

    def test_get_vars_supports_vectors_trajectories_and_scalars(self):
        # Extraction must retain requested group ordering for states and trajectory rows.
        helper = helper_with_fake_ode()
        vector = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
        np.testing.assert_array_equal(helper.get_vars(["velocity", 0], vector), [4, 5, 6, 1])
        self.assertEqual(helper.get_vars("mass", vector, retscalar=True), 7.0)

        trajectory = np.array([vector, vector + 10])
        np.testing.assert_array_equal(helper.get_vars(["mass", "position"], trajectory), [[7, 1, 2, 3], [17, 11, 12, 13]])


if __name__ == "__main__":
    unittest.main()
