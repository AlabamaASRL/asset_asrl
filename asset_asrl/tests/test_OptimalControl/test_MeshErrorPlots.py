"""Unit tests for mesh-error plot construction and OCP delegation."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = PACKAGE_ROOT / "OptimalControl" / "MeshErrorPlots.py"
spec = importlib.util.spec_from_file_location("asset_asrl_mesh_error_plots", MODULE_PATH)
assert spec and spec.loader
mesh_plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_plots)


def sample_phase():
    # A small two-segment mesh exposes plotting behavior without a full optimizer run.
    mesh_iteration = SimpleNamespace(
        times=[0.0, 0.4, 1.0],
        error=[1.0e-3, 1.0e-5],
        distribution=[0.8, 1.2],
        distintegral=[0.0, 0.35, 1.0],
    )
    return SimpleNamespace(MeshTol=1.0e-4, getMeshIters=lambda: [mesh_iteration])


class MeshErrorPlotTests(unittest.TestCase):
    def tearDown(self):
        # Prevent figures from leaking into later tests or interactive sessions.
        plt.close("all")

    def test_phase_mesh_error_plot_creates_expected_axes_and_labels(self):
        # Verify the public plotting contract rather than pixel-level rendering details.
        mesh_plots.PhaseMeshErrorPlot(sample_phase(), show=False)

        figure = plt.gcf()
        self.assertEqual(len(figure.axes), 3)
        self.assertEqual(figure.axes[0].get_ylabel(), "Estimated Error")
        self.assertEqual(figure.axes[1].get_ylabel(), "Error Distribution")
        self.assertEqual(figure.axes[2].get_ylabel(), "Error Distribution Integral")
        self.assertEqual(figure.axes[2].get_xlabel(), "t (0-1)")
        self.assertEqual(figure.axes[0].get_yscale(), "log")
        self.assertEqual(len(figure.axes[0].lines), 2)  # tolerance + iteration error

    def test_ocp_mesh_error_plot_delegates_each_phase_without_showing(self):
        # OCP-level plotting is responsible for visiting every phase exactly once.
        phases = [sample_phase(), sample_phase()]
        ocp = SimpleNamespace(Phases=phases)
        with mock.patch.object(mesh_plots, "PhaseMeshErrorPlot") as phase_plot:
            mesh_plots.OCPMeshErrorPlot(ocp, show=False)

        self.assertEqual(phase_plot.call_args_list, [mock.call(phases[0], show=False), mock.call(phases[1], show=False)])


if __name__ == "__main__":
    unittest.main()
