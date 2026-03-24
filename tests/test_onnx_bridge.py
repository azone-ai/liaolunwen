"""Regression tests for ONNX import/export behavior.

These tests verify that ONNX models can be imported into the internal graph and exported back after fusion.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from fusion_lab.experiments import run_methods
from fusion_lab.hardware import HardwareProfile
from fusion_lab.onnx_bridge import export_fused_onnx, load_onnx_model
from fusion_lab.vendor import ensure_vendor_path

ensure_vendor_path()
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
    onnx.checker.check_model(model)
    onnx.save(model, path)


class OnnxBridgeTests(unittest.TestCase):
    def test_load_onnx_model_converts_to_internal_graph(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'add_relu.onnx'
        _build_add_relu_model(model_path)
        imported = load_onnx_model(model_path)

        self.assertEqual(imported.graph_model.name, 'add_relu_graph')
        self.assertEqual(imported.graph_model.topological_order(), ['add', 'relu'])
        self.assertEqual(imported.graph_model.outputs, ['relu'])
        self.assertIn('x', imported.graph_model.inputs)

    def test_export_fused_onnx_creates_function_backed_fused_node(self) -> None:
        artifact_dir = ROOT / 'outputs' / 'test_artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / 'add_relu.onnx'
        fused_path = artifact_dir / 'add_relu_fused.onnx'
        _build_add_relu_model(model_path)
        imported = load_onnx_model(model_path)
        hardware = HardwareProfile.from_json_file(ROOT / 'configs' / 'hardware' / 'generic_gpu.json')
        results = run_methods(imported.graph_model, hardware, methods=['hw_aware'], max_depth=4)
        export_fused_onnx(imported, results[0], fused_path)

        fused_model = onnx.load(fused_path)
        self.assertEqual(len(fused_model.graph.node), 1)
        self.assertEqual(fused_model.graph.node[0].domain, 'fusion_lab')
        self.assertEqual(len(fused_model.functions), 1)
        self.assertTrue(any(opset.domain == 'fusion_lab' for opset in fused_model.opset_import))
        onnx.checker.check_model(fused_model)


if __name__ == '__main__':
    unittest.main()
