"""Regression tests for reusable fusion-core behavior.

These tests protect mapping classification, legality pruning and high-level search expectations.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from fusion_lab.cost_model import CostModel
from fusion_lab.experiments import run_methods
from fusion_lab.graph import GraphModel
from fusion_lab.hardware import HardwareProfile
from fusion_lab.ops import infer_mapping_type
from fusion_lab.search import optimize_layers_dp, single_layer_plan
from fusion_lab.enums import MappingType


ROOT = Path(__file__).resolve().parents[1]


class FusionPipelineTests(unittest.TestCase):
    def test_reduction_mapping_is_modeled_explicitly(self) -> None:
        graph = GraphModel.from_dict(
            {
                "name": "reduce_case",
                "inputs": [{"name": "x", "shape": [64, 128], "dtype": "float16"}],
                "outputs": ["norm"],
                "nodes": [
                    {
                        "name": "norm",
                        "op": "layer_norm",
                        "inputs": ["x"],
                        "output_shape": [64, 128],
                        "dtype": "float16",
                        "attrs": {"channels": 128, "input_shape": [64, 128]}
                    }
                ]
            }
        )
        self.assertEqual(infer_mapping_type(graph.nodes["norm"]), MappingType.REDUCTION)

    def test_many_to_many_chain_is_pruned(self) -> None:
        graph = GraphModel.from_dict(
            {
                "name": "conv_chain",
                "inputs": [{"name": "x", "shape": [1, 32, 28, 28], "dtype": "float16"}],
                "outputs": ["conv2"],
                "nodes": [
                    {
                        "name": "conv1",
                        "op": "conv2d",
                        "inputs": ["x"],
                        "output_shape": [1, 32, 28, 28],
                        "dtype": "float16",
                        "attrs": {"in_channels": 32, "out_channels": 32, "kernel": [3, 3], "groups": 1}
                    },
                    {
                        "name": "conv2",
                        "op": "conv2d",
                        "inputs": ["conv1"],
                        "output_shape": [1, 32, 28, 28],
                        "dtype": "float16",
                        "attrs": {"in_channels": 32, "out_channels": 32, "kernel": [3, 3], "groups": 1}
                    }
                ]
            }
        )
        hardware = HardwareProfile.from_json_file(ROOT / "configs" / "hardware" / "generic_gpu.json")
        evaluation = CostModel(hardware, enable_soft_penalties=True).evaluate_group(graph, ["conv1", "conv2"])
        self.assertFalse(evaluation.feasible)
        self.assertTrue(any("illegal pairwise fusion" in reason for reason in evaluation.reasons))

    def test_hw_aware_dp_is_not_worse_than_no_fusion_on_sample(self) -> None:
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "residual_block.json")
        hardware = HardwareProfile.from_json_file(ROOT / "configs" / "hardware" / "generic_gpu.json")

        none_plan = single_layer_plan(
            graph,
            CostModel(hardware, enable_soft_penalties=False),
            method="none",
            hardware_name=hardware.name,
        )
        hw_plan = optimize_layers_dp(
            graph,
            CostModel(hardware, enable_soft_penalties=True),
            max_depth=4,
            method="hw_aware",
            hardware_name=hardware.name,
        )

        self.assertTrue(none_plan.feasible)
        self.assertTrue(hw_plan.feasible)
        self.assertGreaterEqual(none_plan.search_time_ms, 0.0)
        self.assertGreaterEqual(hw_plan.search_time_ms, 0.0)
        self.assertGreaterEqual(hw_plan.avg_registers_per_thread, 0.0)
        self.assertGreaterEqual(hw_plan.avg_shared_mem_bytes, 0.0)
        self.assertLessEqual(hw_plan.estimated_latency_ms, none_plan.estimated_latency_ms)

    def test_thread_search_ablation_runs(self) -> None:
        graph = GraphModel.from_json_file(ROOT / "configs" / "graphs" / "residual_block.json")
        hardware = HardwareProfile.from_json_file(ROOT / "configs" / "hardware" / "generic_gpu.json")

        results = run_methods(
            graph,
            hardware,
            methods=["hw_no_thread_search"],
            max_depth=4,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].method, "hw_no_thread_search")
        self.assertTrue(results[0].feasible)
        self.assertGreaterEqual(results[0].search_time_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
