"""Tests for runtime benchmarking and numerical verification helpers.

These tests cover generated inputs, reference verification and optional ONNX Runtime benchmarking.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from fusion_lab.experiments import run_methods
from fusion_lab.hardware import HardwareProfile
from fusion_lab.onnx_bridge import export_fused_onnx, load_onnx_model
from fusion_lab.runtime_benchmark import benchmark_with_onnxruntime, generate_random_inputs, verify_models
from fusion_lab.vendor import ensure_vendor_path

ensure_vendor_path()
import numpy as np
import onnx
from onnx import TensorProto, helper

try:
    import onnxruntime  # type: ignore
    ORT_AVAILABLE = True
except Exception:
    ORT_AVAILABLE = False


ROOT = Path(__file__).resolve().parents[1]


def _build_add_relu_model(path: Path) -> None:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])
    bias = helper.make_tensor('bias', TensorProto.FLOAT, [1, 4], [0.1, 0.2, 0.3, 0.4])
    add = helper.make_node('Add', ['x', 'bias'], ['add_out'], name='add')
    relu = helper.make_node('Relu', ['add_out'], ['y'], name='relu')
    graph = helper.make_graph([add, relu], 'add_relu_graph', [x], [y], initializer=[bias])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    model.ir_version = min(model.ir_version, 11)
    onnx.checker.check_model(model)
    onnx.save(model, path)


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_generate_random_inputs_uses_model_input_specs(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'runtime_add_relu.onnx'
        _build_add_relu_model(model_path)

        feeds, specs = generate_random_inputs(model_path, seed=7)

        self.assertEqual([spec.name for spec in specs], ['x'])
        self.assertEqual(specs[0].shape, [1, 4])
        self.assertEqual(specs[0].dtype, 'float32')
        self.assertIn('x', feeds)
        self.assertEqual(feeds['x'].shape, (1, 4))
        self.assertEqual(feeds['x'].dtype, np.float32)

    def test_reference_verification_matches_original_and_fused_models(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'runtime_add_relu.onnx'
        fused_path = artifact_dir / 'runtime_add_relu_fused.onnx'
        _build_add_relu_model(model_path)

        imported = load_onnx_model(model_path)
        hardware = HardwareProfile.from_json_file(ROOT / 'configs' / 'hardware' / 'generic_gpu.json')
        results = run_methods(imported.graph_model, hardware, methods=['hw_aware'], max_depth=4)
        export_fused_onnx(imported, results[0], fused_path)

        feeds, specs = generate_random_inputs(model_path, seed=11)
        verification = verify_models(
            model_path,
            fused_path,
            feeds,
            specs,
            backend='reference',
            atol=1e-6,
            rtol=1e-6,
        )

        self.assertTrue(verification.allclose)
        self.assertLessEqual(verification.max_abs_diff, 1e-7)
        self.assertEqual(len(verification.outputs), 1)
        self.assertEqual(verification.outputs[0].name, 'y')

    @unittest.skipUnless(ORT_AVAILABLE, 'onnxruntime is not available')
    def test_onnxruntime_benchmark_runs_on_original_and_fused_models(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'runtime_add_relu.onnx'
        fused_path = artifact_dir / 'runtime_add_relu_fused.onnx'
        _build_add_relu_model(model_path)

        imported = load_onnx_model(model_path)
        hardware = HardwareProfile.from_json_file(ROOT / 'configs' / 'hardware' / 'generic_gpu.json')
        results = run_methods(imported.graph_model, hardware, methods=['hw_aware'], max_depth=4)
        export_fused_onnx(imported, results[0], fused_path)

        feeds, specs = generate_random_inputs(model_path, seed=13)
        benchmark = benchmark_with_onnxruntime(
            model_path,
            fused_path,
            feeds,
            specs,
            warmup_runs=1,
            repeat_runs=3,
            providers=['CPUExecutionProvider'],
        )

        self.assertEqual(benchmark.backend, 'onnxruntime')
        self.assertEqual(benchmark.providers, ['CPUExecutionProvider'])
        self.assertEqual(benchmark.original.runs, 3)
        self.assertEqual(benchmark.fused.runs, 3)
        self.assertGreater(benchmark.original.mean_ms, 0.0)
        self.assertGreater(benchmark.fused.mean_ms, 0.0)


if __name__ == '__main__':
    unittest.main()
