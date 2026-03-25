"""Tests for the torch-backed plan execution path."""

from __future__ import annotations

from pathlib import Path
import unittest

from fusion_lab.experiments import run_methods
from fusion_lab.hardware import HardwareProfile
from fusion_lab.onnx_bridge import load_onnx_model
from fusion_lab.research.runtime_validation import generate_random_inputs
from fusion_lab.research.torch_plan_backend import build_torch_plan_module, run_torch_plan_outputs
from fusion_lab.vendor import ensure_vendor_path

ensure_vendor_path()
import numpy as np
import onnx
from onnx import TensorProto, helper


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


def _build_pool_clip_reduce_model(path: Path) -> None:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 1, 1])
    clip_min = helper.make_tensor('clip_min', TensorProto.FLOAT, [], [0.0])
    clip_max = helper.make_tensor('clip_max', TensorProto.FLOAT, [], [2.5])
    axes = helper.make_tensor('axes', TensorProto.INT64, [2], [2, 3])
    avg = helper.make_node(
        'AveragePool',
        ['x'],
        ['avg_out'],
        name='avg_pool',
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        count_include_pad=1,
        ceil_mode=0,
    )
    clip = helper.make_node('Clip', ['avg_out', 'clip_min', 'clip_max'], ['clip_out'], name='clip')
    reduce = helper.make_node('ReduceMean', ['clip_out', 'axes'], ['y'], name='reduce', keepdims=1)
    graph = helper.make_graph([avg, clip, reduce], 'pool_clip_reduce_graph', [x], [y], initializer=[clip_min, clip_max, axes])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
    model.ir_version = min(model.ir_version, 11)
    onnx.checker.check_model(model)
    onnx.save(model, path)


class TorchPlanBackendTests(unittest.TestCase):
    def test_torch_plan_module_matches_expected_output(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'torch_backend_add_relu.onnx'
        _build_add_relu_model(model_path)

        imported = load_onnx_model(model_path)
        hardware = HardwareProfile.from_json_file(ROOT / 'configs' / 'hardware' / 'generic_gpu.json')
        plan = run_methods(imported.graph_model, hardware, methods=['hw_aware'], max_depth=4)[0]

        feeds, _ = generate_random_inputs(model_path, seed=5)
        module = build_torch_plan_module(imported, plan, device='cpu')
        outputs = run_torch_plan_outputs(module, feeds, device='cpu')

        expected = np.maximum(feeds['x'] + np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32), 0.0)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(np.allclose(outputs[0], expected, atol=1e-6, rtol=1e-6))

    def test_torch_plan_module_supports_avgpool_clip_and_reduce_mean(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'torch_backend_pool_clip_reduce.onnx'
        _build_pool_clip_reduce_model(model_path)

        imported = load_onnx_model(model_path)
        hardware = HardwareProfile.from_json_file(ROOT / 'configs' / 'hardware' / 'generic_gpu.json')
        plan = run_methods(imported.graph_model, hardware, methods=['hw_aware'], max_depth=4)[0]

        feeds = {
            'x': np.array([[[[-1.0, 0.5], [3.0, 6.0]]]], dtype=np.float32),
        }
        module = build_torch_plan_module(imported, plan, device='cpu')
        outputs = run_torch_plan_outputs(module, feeds, device='cpu')

        clipped = np.clip(feeds['x'], 0.0, 2.5)
        expected = clipped.mean(axis=(2, 3), keepdims=True)
        self.assertEqual(len(outputs), 1)
        self.assertTrue(np.allclose(outputs[0], expected, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
