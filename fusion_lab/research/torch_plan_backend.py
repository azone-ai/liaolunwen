"""Torch-backed plan execution for more realistic fusion benchmarking.

This module lowers an imported ONNX graph plus a plan result into a PyTorch
module. We insert graph breaks between fusion blocks so `torch.compile`
can optimize within a block while respecting the current fusion partition.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..onnx_bridge import ImportedOnnxModel
from ..search import PlanResult
from ..vendor import import_numpy, import_onnx


def _sanitize_identifier(name: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)
    if not cleaned:
        cleaned = "value"
    if cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return cleaned


def _tensor_to_buffer_name(name: str) -> str:
    return f"buf_{_sanitize_identifier(name)}"


def _numpy_from_initializer(initializer: Any) -> Any:
    onnx = import_onnx()
    return onnx.numpy_helper.to_array(initializer)


def _python_value_from_initializer(initializer: Any) -> Any:
    array = _numpy_from_initializer(initializer)
    if hasattr(array, "tolist"):
        value = array.tolist()
    else:
        value = array
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _node_attrs(node: Any) -> dict[str, Any]:
    onnx = import_onnx()
    attrs: dict[str, Any] = {}
    for attr in node.attribute:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, bytes):
            attrs[attr.name] = value.decode("utf-8", errors="ignore")
        elif isinstance(value, tuple):
            attrs[attr.name] = list(value)
        else:
            attrs[attr.name] = value
    return attrs


def _resolve_reshape_shape(x: torch.Tensor, target_shape: tuple[int, ...]) -> tuple[int, ...]:
    resolved: list[int] = []
    infer_index: int | None = None
    known_product = 1
    input_shape = list(x.shape)
    for idx, dim in enumerate(target_shape):
        if dim == 0:
            value = int(input_shape[idx])
            resolved.append(value)
            known_product *= value
        elif dim == -1:
            infer_index = len(resolved)
            resolved.append(-1)
        else:
            value = int(dim)
            resolved.append(value)
            known_product *= value
    if infer_index is not None:
        total = int(x.numel())
        inferred = total // max(known_product, 1)
        resolved[infer_index] = inferred
    return tuple(resolved)


def _apply_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    dilations: tuple[int, int],
    groups: int,
) -> torch.Tensor:
    if pads != (pads[0], pads[1], pads[0], pads[1]):
        x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]))
        padding = (0, 0)
    else:
        padding = (pads[0], pads[1])
    return F.conv2d(x, weight, bias, stride=strides, padding=padding, dilation=dilations, groups=groups)


def _apply_max_pool2d(
    x: torch.Tensor,
    kernel_shape: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    ceil_mode: bool,
) -> torch.Tensor:
    if pads != (pads[0], pads[1], pads[0], pads[1]):
        x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]), value=float("-inf"))
        padding = (0, 0)
    else:
        padding = (pads[0], pads[1])
    return F.max_pool2d(x, kernel_shape, stride=strides, padding=padding, ceil_mode=ceil_mode)


def _apply_avg_pool2d(
    x: torch.Tensor,
    kernel_shape: tuple[int, int],
    strides: tuple[int, int],
    pads: tuple[int, int, int, int],
    ceil_mode: bool,
    count_include_pad: bool,
) -> torch.Tensor:
    if pads != (pads[0], pads[1], pads[0], pads[1]):
        x = F.pad(x, (pads[1], pads[3], pads[0], pads[2]), value=0.0)
        padding = (0, 0)
    else:
        padding = (pads[0], pads[1])
    return F.avg_pool2d(
        x,
        kernel_shape,
        stride=strides,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )


def _apply_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None,
    alpha: float,
    beta: float,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    if trans_a:
        a = a.transpose(-1, -2)
    if trans_b:
        b = b.transpose(-1, -2)
    y = torch.matmul(a, b)
    if alpha != 1.0:
        y = y * alpha
    if c is not None:
        bias = c
        if beta != 1.0:
            bias = bias * beta
        y = y + bias
    return y


def _coerce_optional_bound(value: torch.Tensor | float | int | None) -> torch.Tensor | float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        if value.numel() == 1:
            return float(value.detach().item())
        return value
    return float(value)


def _apply_clip(
    x: torch.Tensor,
    min_value: torch.Tensor | float | int | None = None,
    max_value: torch.Tensor | float | int | None = None,
) -> torch.Tensor:
    min_arg = _coerce_optional_bound(min_value)
    max_arg = _coerce_optional_bound(max_value)
    if min_arg is None and max_arg is None:
        return x
    return torch.clamp(x, min=min_arg, max=max_arg)


def _coerce_axes(axes: torch.Tensor | list[int] | tuple[int, ...] | int | None) -> tuple[int, ...] | None:
    if axes is None:
        return None
    if isinstance(axes, torch.Tensor):
        if axes.numel() == 0:
            return ()
        values = axes.detach().cpu().tolist()
        if isinstance(values, list):
            return tuple(int(value) for value in values)
        return (int(values),)
    if isinstance(axes, (list, tuple)):
        return tuple(int(value) for value in axes)
    return (int(axes),)


def _apply_reduce_mean(
    x: torch.Tensor,
    axes: torch.Tensor | list[int] | tuple[int, ...] | int | None,
    keepdim: bool,
    noop_with_empty_axes: bool,
) -> torch.Tensor:
    dims = _coerce_axes(axes)
    if dims is None:
        return x if noop_with_empty_axes else torch.mean(x)
    if len(dims) == 0:
        return x if noop_with_empty_axes else torch.mean(x)
    rank = x.dim()
    normalized_dims = tuple(sorted({dim if dim >= 0 else rank + dim for dim in dims}))
    return torch.mean(x, dim=normalized_dims, keepdim=keepdim)


def _tensor_from_numpy(array: Any, device: torch.device) -> torch.Tensor:
    np = import_numpy()
    tensor = torch.from_numpy(np.asarray(array).copy())
    if tensor.dtype == torch.float64:
        tensor = tensor.float()
    return tensor.to(device=device)


@dataclass(frozen=True)
class TorchBackendStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    runs: int
    compile_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "runs": self.runs,
            "compile_time_ms": self.compile_time_ms,
        }


@dataclass(frozen=True)
class TorchVerification:
    allclose: bool
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "allclose": self.allclose,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "cosine_similarity": self.cosine_similarity,
        }


@dataclass
class TorchPlanExecutor:
    mode: str
    device: torch.device
    graph_input_names: list[str]
    graph_output_names: list[str]
    module: nn.Module | None = None
    block_modules: list[nn.Module] | None = None
    block_input_names: list[list[str]] | None = None
    block_output_names: list[list[str]] | None = None


class TorchPlanModule(nn.Module):
    pass


def build_torch_plan_module(
    imported: ImportedOnnxModel,
    plan: PlanResult,
    device: str | torch.device = "cpu",
) -> nn.Module:
    device = torch.device(device)
    initializer_map = {initializer.name: initializer for initializer in imported.model.graph.initializer}
    graph_input_names = [
        value_info.name
        for value_info in imported.model.graph.input
        if value_info.name not in initializer_map
    ]
    block_order_by_node = {
        node_name: block_idx
        for block_idx, block in enumerate(plan.blocks)
        for node_name in block.nodes
    }
    node_output_tensor = imported.node_name_to_output_tensor

    source_lines = [
        "def forward(self, " + ", ".join(_sanitize_identifier(name) for name in graph_input_names) + "):"
    ]
    if not graph_input_names:
        source_lines = ["def forward(self):"]

    env: dict[str, str] = {name: _sanitize_identifier(name) for name in graph_input_names}
    previous_block_idx: int | None = None
    buffer_map: dict[str, str] = {}

    def tensor_expr(tensor_name: str) -> str:
        if tensor_name in env:
            return env[tensor_name]
        if tensor_name in initializer_map:
            if tensor_name not in buffer_map:
                buffer_map[tensor_name] = _tensor_to_buffer_name(tensor_name)
            return f"self.{buffer_map[tensor_name]}"
        raise KeyError(f"Tensor '{tensor_name}' is not available in the generated environment.")

    for node_name in imported.ordered_node_names:
        proto = imported.node_name_to_proto[node_name]
        block_idx = block_order_by_node.get(node_name, -1)
        if previous_block_idx is not None and block_idx != previous_block_idx:
            source_lines.append("    torch._dynamo.graph_break()")
        previous_block_idx = block_idx

        attrs = _node_attrs(proto)
        op_type = proto.op_type
        output_tensor = node_output_tensor[node_name]
        output_var = _sanitize_identifier(output_tensor)
        non_empty_inputs = [name for name in proto.input if name]

        if op_type == "Add":
            a, b = (tensor_expr(non_empty_inputs[0]), tensor_expr(non_empty_inputs[1]))
            source_lines.append(f"    {output_var} = {a} + {b}")
        elif op_type == "Relu":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = F.relu({x})")
        elif op_type == "Conv":
            x = tensor_expr(non_empty_inputs[0])
            weight = tensor_expr(non_empty_inputs[1])
            bias = tensor_expr(non_empty_inputs[2]) if len(non_empty_inputs) > 2 else "None"
            strides = tuple(int(v) for v in attrs.get("strides", [1, 1]))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            dilations = tuple(int(v) for v in attrs.get("dilations", [1, 1]))
            groups = int(attrs.get("group", 1))
            source_lines.append(
                f"    {output_var} = _apply_conv2d({x}, {weight}, {bias}, {strides}, {pads}, {dilations}, {groups})"
            )
        elif op_type == "BatchNormalization":
            x = tensor_expr(non_empty_inputs[0])
            scale = tensor_expr(non_empty_inputs[1])
            bias = tensor_expr(non_empty_inputs[2])
            mean = tensor_expr(non_empty_inputs[3])
            var = tensor_expr(non_empty_inputs[4])
            eps = float(attrs.get("epsilon", 1e-5))
            source_lines.append(
                f"    {output_var} = F.batch_norm({x}, {mean}, {var}, {scale}, {bias}, training=False, momentum=0.1, eps={eps})"
            )
        elif op_type == "MaxPool":
            x = tensor_expr(non_empty_inputs[0])
            kernel = tuple(int(v) for v in attrs.get("kernel_shape", [1, 1]))
            strides = tuple(int(v) for v in attrs.get("strides", kernel))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            ceil_mode = bool(int(attrs.get("ceil_mode", 0)))
            source_lines.append(
                f"    {output_var} = _apply_max_pool2d({x}, {kernel}, {strides}, {pads}, {ceil_mode})"
            )
        elif op_type == "AveragePool":
            x = tensor_expr(non_empty_inputs[0])
            kernel = tuple(int(v) for v in attrs.get("kernel_shape", [1, 1]))
            strides = tuple(int(v) for v in attrs.get("strides", kernel))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            ceil_mode = bool(int(attrs.get("ceil_mode", 0)))
            count_include_pad = bool(int(attrs.get("count_include_pad", 0)))
            source_lines.append(
                f"    {output_var} = _apply_avg_pool2d({x}, {kernel}, {strides}, {pads}, {ceil_mode}, {count_include_pad})"
            )
        elif op_type == "GlobalAveragePool":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = {x}.mean(dim=(-1, -2), keepdim=True)")
        elif op_type == "Clip":
            x = tensor_expr(non_empty_inputs[0])
            min_expr = "None"
            max_expr = "None"
            if len(non_empty_inputs) > 1:
                if non_empty_inputs[1] in initializer_map:
                    min_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[1]]))
                else:
                    min_expr = tensor_expr(non_empty_inputs[1])
            elif "min" in attrs:
                min_expr = repr(float(attrs["min"]))
            if len(non_empty_inputs) > 2:
                if non_empty_inputs[2] in initializer_map:
                    max_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[2]]))
                else:
                    max_expr = tensor_expr(non_empty_inputs[2])
            elif "max" in attrs:
                max_expr = repr(float(attrs["max"]))
            source_lines.append(f"    {output_var} = _apply_clip({x}, {min_expr}, {max_expr})")
        elif op_type == "Flatten":
            x = tensor_expr(non_empty_inputs[0])
            axis = int(attrs.get("axis", 1))
            source_lines.append(f"    {output_var} = torch.flatten({x}, start_dim={axis})")
        elif op_type == "Reshape":
            x = tensor_expr(non_empty_inputs[0])
            shape_input = non_empty_inputs[1]
            if shape_input in initializer_map:
                target = tuple(int(v) for v in _numpy_from_initializer(initializer_map[shape_input]).tolist())
                source_lines.append(f"    {output_var} = torch.reshape({x}, _resolve_reshape_shape({x}, {target}))")
            else:
                shape_expr = tensor_expr(shape_input)
                source_lines.append(
                    f"    {output_var} = torch.reshape({x}, tuple(int(v) for v in {shape_expr}.tolist()))"
                )
        elif op_type == "Gemm":
            a = tensor_expr(non_empty_inputs[0])
            b = tensor_expr(non_empty_inputs[1])
            c = tensor_expr(non_empty_inputs[2]) if len(non_empty_inputs) > 2 else "None"
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            trans_a = bool(int(attrs.get("transA", 0)))
            trans_b = bool(int(attrs.get("transB", 0)))
            source_lines.append(
                f"    {output_var} = _apply_gemm({a}, {b}, {c}, {alpha}, {beta}, {trans_a}, {trans_b})"
            )
        elif op_type == "Dropout":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = {x}")
        elif op_type == "ReduceMean":
            x = tensor_expr(non_empty_inputs[0])
            keepdims = bool(int(attrs.get("keepdims", 1)))
            noop_with_empty_axes = bool(int(attrs.get("noop_with_empty_axes", 0)))
            if len(non_empty_inputs) > 1:
                if non_empty_inputs[1] in initializer_map:
                    axes_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[1]]))
                else:
                    axes_expr = tensor_expr(non_empty_inputs[1])
            elif "axes" in attrs:
                axes_expr = repr(tuple(int(v) for v in attrs.get("axes", [])))
            else:
                axes_expr = "None"
            source_lines.append(
                f"    {output_var} = _apply_reduce_mean({x}, {axes_expr}, {keepdims}, {noop_with_empty_axes})"
            )
        else:
            raise NotImplementedError(f"Unsupported op for torch backend: {op_type}")

        env[output_tensor] = output_var

    graph_outputs = [value_info.name for value_info in imported.model.graph.output]
    output_exprs = [env[name] for name in graph_outputs]
    if len(output_exprs) == 1:
        source_lines.append(f"    return {output_exprs[0]}")
    else:
        source_lines.append(f"    return ({', '.join(output_exprs)})")

    namespace: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "_apply_conv2d": _apply_conv2d,
        "_apply_max_pool2d": _apply_max_pool2d,
        "_apply_avg_pool2d": _apply_avg_pool2d,
        "_apply_gemm": _apply_gemm,
        "_apply_clip": _apply_clip,
        "_apply_reduce_mean": _apply_reduce_mean,
        "_resolve_reshape_shape": _resolve_reshape_shape,
    }
    local_ns: dict[str, Any] = {}
    exec("\n".join(source_lines), namespace, local_ns)
    forward_fn = local_ns["forward"]

    module = TorchPlanModule()
    for tensor_name, buffer_name in buffer_map.items():
        array = _numpy_from_initializer(initializer_map[tensor_name])
        module.register_buffer(buffer_name, _tensor_from_numpy(array, device=device), persistent=False)
    module.forward = forward_fn.__get__(module, TorchPlanModule)
    module.to(device)
    module.eval()
    return module


def _block_external_inputs(imported: ImportedOnnxModel, node_names: list[str]) -> list[str]:
    initializer_names = {initializer.name for initializer in imported.model.graph.initializer}
    group = set(node_names)
    ordered_inputs: list[str] = []
    seen: set[str] = set()
    for node_name in node_names:
        node = imported.node_name_to_proto[node_name]
        for tensor_name in node.input:
            if not tensor_name or tensor_name in initializer_names:
                continue
            producer = imported.output_tensor_to_node_name.get(tensor_name)
            if producer not in group and tensor_name not in seen:
                seen.add(tensor_name)
                ordered_inputs.append(tensor_name)
    return ordered_inputs


def _block_external_outputs(imported: ImportedOnnxModel, node_names: list[str]) -> list[str]:
    group = set(node_names)
    ordered_outputs: list[str] = []
    for node_name in node_names:
        tensor_name = imported.node_name_to_output_tensor[node_name]
        consumers = imported.tensor_consumers.get(tensor_name, [])
        if tensor_name in imported.graph_output_tensors or any(consumer not in group for consumer in consumers):
            ordered_outputs.append(tensor_name)
    return ordered_outputs


def build_torch_block_module(
    imported: ImportedOnnxModel,
    node_names: list[str],
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, list[str], list[str]]:
    device = torch.device(device)
    initializer_map = {initializer.name: initializer for initializer in imported.model.graph.initializer}
    block_inputs = _block_external_inputs(imported, node_names)
    block_outputs = _block_external_outputs(imported, node_names)
    node_output_tensor = imported.node_name_to_output_tensor

    source_lines = [
        "def forward(self, " + ", ".join(_sanitize_identifier(name) for name in block_inputs) + "):"
    ]
    if not block_inputs:
        source_lines = ["def forward(self):"]

    env: dict[str, str] = {name: _sanitize_identifier(name) for name in block_inputs}
    buffer_map: dict[str, str] = {}

    def tensor_expr(tensor_name: str) -> str:
        if tensor_name in env:
            return env[tensor_name]
        if tensor_name in initializer_map:
            if tensor_name not in buffer_map:
                buffer_map[tensor_name] = _tensor_to_buffer_name(tensor_name)
            return f"self.{buffer_map[tensor_name]}"
        raise KeyError(f"Tensor '{tensor_name}' is not available in the generated block environment.")

    for node_name in node_names:
        proto = imported.node_name_to_proto[node_name]
        attrs = _node_attrs(proto)
        op_type = proto.op_type
        output_tensor = node_output_tensor[node_name]
        output_var = _sanitize_identifier(output_tensor)
        non_empty_inputs = [name for name in proto.input if name]

        if op_type == "Add":
            a, b = (tensor_expr(non_empty_inputs[0]), tensor_expr(non_empty_inputs[1]))
            source_lines.append(f"    {output_var} = {a} + {b}")
        elif op_type == "Relu":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = F.relu({x})")
        elif op_type == "Conv":
            x = tensor_expr(non_empty_inputs[0])
            weight = tensor_expr(non_empty_inputs[1])
            bias = tensor_expr(non_empty_inputs[2]) if len(non_empty_inputs) > 2 else "None"
            strides = tuple(int(v) for v in attrs.get("strides", [1, 1]))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            dilations = tuple(int(v) for v in attrs.get("dilations", [1, 1]))
            groups = int(attrs.get("group", 1))
            source_lines.append(
                f"    {output_var} = _apply_conv2d({x}, {weight}, {bias}, {strides}, {pads}, {dilations}, {groups})"
            )
        elif op_type == "BatchNormalization":
            x = tensor_expr(non_empty_inputs[0])
            scale = tensor_expr(non_empty_inputs[1])
            bias = tensor_expr(non_empty_inputs[2])
            mean = tensor_expr(non_empty_inputs[3])
            var = tensor_expr(non_empty_inputs[4])
            eps = float(attrs.get("epsilon", 1e-5))
            source_lines.append(
                f"    {output_var} = F.batch_norm({x}, {mean}, {var}, {scale}, {bias}, training=False, momentum=0.1, eps={eps})"
            )
        elif op_type == "MaxPool":
            x = tensor_expr(non_empty_inputs[0])
            kernel = tuple(int(v) for v in attrs.get("kernel_shape", [1, 1]))
            strides = tuple(int(v) for v in attrs.get("strides", kernel))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            ceil_mode = bool(int(attrs.get("ceil_mode", 0)))
            source_lines.append(
                f"    {output_var} = _apply_max_pool2d({x}, {kernel}, {strides}, {pads}, {ceil_mode})"
            )
        elif op_type == "AveragePool":
            x = tensor_expr(non_empty_inputs[0])
            kernel = tuple(int(v) for v in attrs.get("kernel_shape", [1, 1]))
            strides = tuple(int(v) for v in attrs.get("strides", kernel))
            pads = tuple(int(v) for v in attrs.get("pads", [0, 0, 0, 0]))
            ceil_mode = bool(int(attrs.get("ceil_mode", 0)))
            count_include_pad = bool(int(attrs.get("count_include_pad", 0)))
            source_lines.append(
                f"    {output_var} = _apply_avg_pool2d({x}, {kernel}, {strides}, {pads}, {ceil_mode}, {count_include_pad})"
            )
        elif op_type == "GlobalAveragePool":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = {x}.mean(dim=(-1, -2), keepdim=True)")
        elif op_type == "Clip":
            x = tensor_expr(non_empty_inputs[0])
            min_expr = "None"
            max_expr = "None"
            if len(non_empty_inputs) > 1:
                if non_empty_inputs[1] in initializer_map:
                    min_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[1]]))
                else:
                    min_expr = tensor_expr(non_empty_inputs[1])
            elif "min" in attrs:
                min_expr = repr(float(attrs["min"]))
            if len(non_empty_inputs) > 2:
                if non_empty_inputs[2] in initializer_map:
                    max_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[2]]))
                else:
                    max_expr = tensor_expr(non_empty_inputs[2])
            elif "max" in attrs:
                max_expr = repr(float(attrs["max"]))
            source_lines.append(f"    {output_var} = _apply_clip({x}, {min_expr}, {max_expr})")
        elif op_type == "Flatten":
            x = tensor_expr(non_empty_inputs[0])
            axis = int(attrs.get("axis", 1))
            source_lines.append(f"    {output_var} = torch.flatten({x}, start_dim={axis})")
        elif op_type == "Reshape":
            x = tensor_expr(non_empty_inputs[0])
            shape_input = non_empty_inputs[1]
            if shape_input in initializer_map:
                target = tuple(int(v) for v in _numpy_from_initializer(initializer_map[shape_input]).tolist())
                source_lines.append(f"    {output_var} = torch.reshape({x}, _resolve_reshape_shape({x}, {target}))")
            else:
                shape_expr = tensor_expr(shape_input)
                source_lines.append(
                    f"    {output_var} = torch.reshape({x}, tuple(int(v) for v in {shape_expr}.tolist()))"
                )
        elif op_type == "Gemm":
            a = tensor_expr(non_empty_inputs[0])
            b = tensor_expr(non_empty_inputs[1])
            c = tensor_expr(non_empty_inputs[2]) if len(non_empty_inputs) > 2 else "None"
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            trans_a = bool(int(attrs.get("transA", 0)))
            trans_b = bool(int(attrs.get("transB", 0)))
            source_lines.append(
                f"    {output_var} = _apply_gemm({a}, {b}, {c}, {alpha}, {beta}, {trans_a}, {trans_b})"
            )
        elif op_type == "Dropout":
            x = tensor_expr(non_empty_inputs[0])
            source_lines.append(f"    {output_var} = {x}")
        elif op_type == "ReduceMean":
            x = tensor_expr(non_empty_inputs[0])
            keepdims = bool(int(attrs.get("keepdims", 1)))
            noop_with_empty_axes = bool(int(attrs.get("noop_with_empty_axes", 0)))
            if len(non_empty_inputs) > 1:
                if non_empty_inputs[1] in initializer_map:
                    axes_expr = repr(_python_value_from_initializer(initializer_map[non_empty_inputs[1]]))
                else:
                    axes_expr = tensor_expr(non_empty_inputs[1])
            elif "axes" in attrs:
                axes_expr = repr(tuple(int(v) for v in attrs.get("axes", [])))
            else:
                axes_expr = "None"
            source_lines.append(
                f"    {output_var} = _apply_reduce_mean({x}, {axes_expr}, {keepdims}, {noop_with_empty_axes})"
            )
        else:
            raise NotImplementedError(f"Unsupported op for torch block backend: {op_type}")

        env[output_tensor] = output_var

    output_exprs = [env[name] for name in block_outputs]
    if len(output_exprs) == 1:
        source_lines.append(f"    return {output_exprs[0]}")
    else:
        source_lines.append(f"    return ({', '.join(output_exprs)})")

    namespace: dict[str, Any] = {
        "torch": torch,
        "F": F,
        "_apply_conv2d": _apply_conv2d,
        "_apply_max_pool2d": _apply_max_pool2d,
        "_apply_avg_pool2d": _apply_avg_pool2d,
        "_apply_gemm": _apply_gemm,
        "_apply_clip": _apply_clip,
        "_apply_reduce_mean": _apply_reduce_mean,
        "_resolve_reshape_shape": _resolve_reshape_shape,
    }
    local_ns: dict[str, Any] = {}
    exec("\n".join(source_lines), namespace, local_ns)
    forward_fn = local_ns["forward"]

    module = TorchPlanModule()
    for tensor_name, buffer_name in buffer_map.items():
        array = _numpy_from_initializer(initializer_map[tensor_name])
        module.register_buffer(buffer_name, _tensor_from_numpy(array, device=device), persistent=False)
    module.forward = forward_fn.__get__(module, TorchPlanModule)
    module.to(device)
    module.eval()
    return module, block_inputs, block_outputs


def _prepare_inputs(feeds: dict[str, Any], device: str | torch.device) -> tuple[list[str], list[torch.Tensor]]:
    device = torch.device(device)
    ordered_names = list(feeds.keys())
    tensors = [_tensor_from_numpy(feeds[name], device) for name in ordered_names]
    return ordered_names, tensors


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_torch_plan(
    imported: ImportedOnnxModel,
    plan: PlanResult,
    feeds: dict[str, Any],
    device: str | torch.device = "cuda",
    warmup_runs: int = 10,
    repeat_runs: int = 30,
    compile_model: bool = True,
    backend: str = "auto",
) -> tuple[TorchBackendStats, TorchPlanExecutor]:
    device = torch.device(device)
    graph_input_names = [
        value_info.name
        for value_info in imported.model.graph.input
        if value_info.name not in {initializer.name for initializer in imported.model.graph.initializer}
    ]
    graph_output_names = [value_info.name for value_info in imported.model.graph.output]

    selected_backend = backend
    if selected_backend == "auto":
        selected_backend = "inductor" if device.type == "cuda" else "trace"

    if selected_backend == "inductor":
        input_names, input_tensors = _prepare_inputs(feeds, device)
        module = build_torch_plan_module(imported, plan, device=device)
        compile_start = perf_counter()
        if compile_model:
            try:
                module = torch.compile(module, backend="inductor", fullgraph=False, mode="reduce-overhead")
            except Exception:
                selected_backend = "trace"
        if selected_backend == "inductor":
            with torch.inference_mode():
                _ = module(*input_tensors)
                _sync_device(device)
            compile_time_ms = (perf_counter() - compile_start) * 1000.0
            executor = TorchPlanExecutor(
                mode="inductor",
                device=device,
                graph_input_names=graph_input_names,
                graph_output_names=graph_output_names,
                module=module,
            )
        else:
            compile_time_ms = 0.0
    else:
        compile_time_ms = 0.0

    if selected_backend == "trace":
        block_modules: list[nn.Module] = []
        block_input_names: list[list[str]] = []
        block_output_names: list[list[str]] = []
        tensor_env = {name: _tensor_from_numpy(value, device=device) for name, value in feeds.items()}
        for block in plan.blocks:
            block_module, inputs_for_block, outputs_for_block = build_torch_block_module(
                imported,
                block.nodes,
                device=device,
            )
            example_inputs = tuple(tensor_env[name] for name in inputs_for_block)
            trace_start = perf_counter()
            traced = torch.jit.trace(block_module, example_inputs, check_trace=False)
            traced_outputs = traced(*example_inputs)
            _sync_device(device)
            compile_time_ms += (perf_counter() - trace_start) * 1000.0
            if not isinstance(traced_outputs, tuple):
                traced_outputs = (traced_outputs,)
            for output_name, value in zip(outputs_for_block, traced_outputs):
                tensor_env[output_name] = value
            block_modules.append(traced)
            block_input_names.append(inputs_for_block)
            block_output_names.append(outputs_for_block)
        executor = TorchPlanExecutor(
            mode="trace",
            device=device,
            graph_input_names=graph_input_names,
            graph_output_names=graph_output_names,
            block_modules=block_modules,
            block_input_names=block_input_names,
            block_output_names=block_output_names,
        )

    with torch.inference_mode():
        for _ in range(max(warmup_runs, 0)):
            _ = _run_executor(executor, feeds)
        _sync_device(device)

        timings_ms: list[float] = []
        for _ in range(max(repeat_runs, 1)):
            start = perf_counter()
            _ = _run_executor(executor, feeds)
            _sync_device(device)
            timings_ms.append((perf_counter() - start) * 1000.0)

    np = import_numpy()
    values = np.asarray(timings_ms, dtype=np.float64)
    stats = TorchBackendStats(
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        std_ms=float(values.std()),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
        runs=int(values.size),
        compile_time_ms=compile_time_ms,
    )
    return stats, executor


def run_torch_plan_outputs(
    executor: TorchPlanExecutor | nn.Module,
    feeds: dict[str, Any],
    device: str | torch.device = "cuda",
) -> list[Any]:
    if isinstance(executor, nn.Module):
        device = torch.device(device)
        _, input_tensors = _prepare_inputs(feeds, device)
        with torch.inference_mode():
            outputs = executor(*input_tensors)
            _sync_device(device)
        if isinstance(outputs, tuple):
            result = list(outputs)
        else:
            result = [outputs]
        return [output.detach().cpu().numpy() for output in result]

    with torch.inference_mode():
        outputs = _run_executor(executor, feeds)
        _sync_device(torch.device(device))
    if isinstance(outputs, tuple):
        result = list(outputs)
    else:
        result = [outputs]
    return [output.detach().cpu().numpy() for output in result]


def _run_executor(executor: TorchPlanExecutor, feeds: dict[str, Any]) -> Any:
    if executor.mode == "inductor":
        _, input_tensors = _prepare_inputs(feeds, executor.device)
        assert executor.module is not None
        return executor.module(*input_tensors)

    assert executor.block_modules is not None
    assert executor.block_input_names is not None
    assert executor.block_output_names is not None

    tensor_env = {
        name: _tensor_from_numpy(value, device=executor.device)
        for name, value in feeds.items()
    }
    for module, input_names, output_names in zip(
        executor.block_modules,
        executor.block_input_names,
        executor.block_output_names,
    ):
        args = [tensor_env[name] for name in input_names]
        outputs = module(*args)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for output_name, value in zip(output_names, outputs):
            tensor_env[output_name] = value

    final_outputs = [tensor_env[name] for name in executor.graph_output_names]
    if len(final_outputs) == 1:
        return final_outputs[0]
    return tuple(final_outputs)


def verify_torch_plan_against_onnx(
    original_outputs: list[Any],
    candidate_outputs: list[Any],
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> TorchVerification:
    np = import_numpy()
    if len(original_outputs) != len(candidate_outputs):
        raise ValueError("Output count mismatch between reference and candidate plan.")
    max_abs_diff = 0.0
    total_abs = 0.0
    total_elems = 0
    allclose = True
    cosine_values: list[float] = []
    for baseline, candidate in zip(original_outputs, candidate_outputs):
        base = np.asarray(baseline)
        cand = np.asarray(candidate)
        abs_diff = np.abs(base - cand)
        max_abs_diff = max(max_abs_diff, float(abs_diff.max()) if abs_diff.size else 0.0)
        total_abs += float(abs_diff.sum()) if abs_diff.size else 0.0
        total_elems += int(abs_diff.size)
        allclose = allclose and bool(np.allclose(base, cand, atol=atol, rtol=rtol, equal_nan=True))
        base_flat = base.astype(np.float64).reshape(-1)
        cand_flat = cand.astype(np.float64).reshape(-1)
        base_norm = float(np.linalg.norm(base_flat))
        cand_norm = float(np.linalg.norm(cand_flat))
        if base_norm == 0.0 and cand_norm == 0.0:
            cosine_values.append(1.0)
        elif base_norm == 0.0 or cand_norm == 0.0:
            cosine_values.append(0.0)
        else:
            cosine_values.append(float(np.dot(base_flat, cand_flat) / (base_norm * cand_norm)))
    mean_abs_diff = total_abs / total_elems if total_elems > 0 else 0.0
    cosine_similarity = sum(cosine_values) / len(cosine_values) if cosine_values else 1.0
    return TorchVerification(
        allclose=allclose,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        cosine_similarity=cosine_similarity,
    )


def write_torch_backend_report(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    include_model = any("model_name" in row for row in rows)
    markdown_lines = [
        "# Torch Plan Backend Benchmark",
        "",
        (
            "| Model | Method | Mean (ms) | Speedup | Compile (ms) | Search (ms) | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Allclose | Max Abs Diff |"
            if include_model
            else "| Method | Mean (ms) | Speedup | Compile (ms) | Search (ms) | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Allclose | Max Abs Diff |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
            if include_model
            else "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"
        ),
    ]
    for row in rows:
        speedup_text = "-" if row.get("runtime_speedup") is None else f"{row['runtime_speedup']:.4f}"
        if include_model:
            markdown_lines.append(
                f"| {row.get('model_name', '-')} | {row['method']} | {row['runtime_mean_ms']:.4f} | {speedup_text} | {row['compile_time_ms']:.2f} | {row['search_time_ms']:.3f} | {row['avg_occupancy']:.3f} | {row['avg_registers_per_thread']:.2f} | {row['avg_shared_mem_kib']:.3f} | {row['allclose']} | {row['max_abs_diff']:.8f} |"
            )
        else:
            markdown_lines.append(
                f"| {row['method']} | {row['runtime_mean_ms']:.4f} | {speedup_text} | {row['compile_time_ms']:.2f} | {row['search_time_ms']:.3f} | {row['avg_occupancy']:.3f} | {row['avg_registers_per_thread']:.2f} | {row['avg_shared_mem_kib']:.3f} | {row['allclose']} | {row['max_abs_diff']:.8f} |"
            )
    output_path.with_suffix(".md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return output_path
