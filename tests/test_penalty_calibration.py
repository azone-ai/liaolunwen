"""Tests for penalty calibration profiles and auto-selection."""

from __future__ import annotations

from pathlib import Path
import unittest

from fusion_lab.graph import GraphModel
from fusion_lab.hardware import HardwareProfile
from fusion_lab.research.penalty_calibration import (
    apply_penalty_profile,
    choose_profile_for_graph,
    summarize_graph_features,
)


ROOT = Path(__file__).resolve().parents[1]


class PenaltyCalibrationTests(unittest.TestCase):
    def test_global_soft_profile_relaxes_penalties(self) -> None:
        hardware = HardwareProfile.from_json_file(ROOT / "configs" / "hardware" / "generic_gpu.json")
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "residual_block.json")

        calibrated, profile_name, _ = apply_penalty_profile(hardware, graph, requested_profile="global_soft")

        self.assertEqual(profile_name, "global_soft")
        self.assertEqual(calibrated.name, "generic_gpu:global_soft")
        self.assertAlmostEqual(calibrated.penalty_weights["register"], 0.18)
        self.assertAlmostEqual(calibrated.penalty_weights["shared_memory"], 0.12)
        self.assertAlmostEqual(calibrated.penalty_weights["geometry"], 0.10)
        self.assertAlmostEqual(calibrated.penalty_weights["icache"], 0.06)
        self.assertAlmostEqual(calibrated.reg_soft_ratio, 0.60)
        self.assertAlmostEqual(calibrated.smem_soft_ratio, 0.65)

    def test_auto_profile_detects_depthwise_style_graph(self) -> None:
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "mobilenet_bottleneck.json")

        profile_name, summary = choose_profile_for_graph(graph)

        self.assertEqual(profile_name, "cnn_light")
        self.assertGreater(summary.depthwise_ratio, 0.0)

    def test_auto_profile_detects_residual_graph(self) -> None:
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "residual_block.json")

        profile_name, summary = choose_profile_for_graph(graph)

        self.assertEqual(profile_name, "cnn_residual")
        self.assertGreater(summary.residual_add_ratio, 0.0)

    def test_auto_profile_detects_reduction_graph(self) -> None:
        graph = GraphModel.from_dict(
            {
                "name": "reduction_case",
                "inputs": [{"name": "x", "shape": [1, 16, 64], "dtype": "float32"}],
                "outputs": ["out"],
                "nodes": [
                    {
                        "name": "mean0",
                        "op": "reduce_mean",
                        "inputs": ["x"],
                        "output_shape": [1, 16, 1],
                        "dtype": "float32",
                        "attrs": {"input_shape": [1, 16, 64]},
                    },
                    {
                        "name": "out",
                        "op": "layer_norm",
                        "inputs": ["mean0"],
                        "output_shape": [1, 16, 1],
                        "dtype": "float32",
                        "attrs": {"channels": 16, "input_shape": [1, 16, 1]},
                    },
                ],
            }
        )

        profile_name, summary = choose_profile_for_graph(graph)

        self.assertEqual(profile_name, "reduction_tf")
        self.assertGreaterEqual(summary.reduction_ratio, 0.5)

    def test_feature_summary_counts_ops(self) -> None:
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "residual_block.json")

        summary = summarize_graph_features(graph)

        self.assertEqual(summary.total_nodes, len(graph.nodes))
        self.assertIn("add", summary.op_counts)


if __name__ == "__main__":
    unittest.main()
