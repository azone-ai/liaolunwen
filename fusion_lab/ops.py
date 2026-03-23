from __future__ import annotations

from typing import Any

from .enums import MappingType
from .graph import NodeSpec, dtype_nbytes, num_elements


ONE_TO_ONE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "clip",
    "batch_norm",
    "bias_add",
}

ONE_TO_MANY_OPS = {
    "expand",
    "resize",
    "upsample",
    "gather",
    "broadcast_to",
}

MANY_TO_MANY_OPS = {
    "conv1d",
    "conv2d",
    "conv3d",
    "depthwise_conv2d",
    "gemm",
    "matmul",
    "avg_pool2d",
    "max_pool2d",
}

REORGANIZE_OPS = {
    "reshape",
    "flatten",
    "squeeze",
    "unsqueeze",
}

SHUFFLE_OPS = {
    "transpose",
    "permute",
    "depth_to_space",
    "space_to_depth",
}

REDUCTION_OPS = {
    "reduce_sum",
    "reduce_mean",
    "reduce_max",
    "reduce_min",
    "layer_norm",
}

OPAQUE_OPS = {
    "softmax",
    "concat",
    "split",
    "sort",
    "topk",
}


ONNX_ALIASES = {
    'conv': 'conv2d',
    'convtranspose': 'conv2d',
    'batchnormalization': 'batch_norm',
    'maxpool': 'max_pool2d',
    'averagepool': 'avg_pool2d',
    'globalaveragepool': 'avg_pool2d',
    'globalmaxpool': 'max_pool2d',
    'layernormalization': 'layer_norm',
    'reducesum': 'reduce_sum',
    'reducemean': 'reduce_mean',
    'reducemax': 'reduce_max',
    'reducemin': 'reduce_min',
}


def normalize_op(op_type: str) -> str:
    op = op_type.strip().lower()
    return ONNX_ALIASES.get(op, op)


def infer_mapping_type(node: NodeSpec) -> MappingType:
    if node.mapping_type:
        return MappingType(node.mapping_type)
    op = normalize_op(node.op_type)
    if op in ONE_TO_ONE_OPS:
        return MappingType.ONE_TO_ONE
    if op in ONE_TO_MANY_OPS:
        return MappingType.ONE_TO_MANY
    if op in MANY_TO_MANY_OPS:
        return MappingType.MANY_TO_MANY
    if op in REORGANIZE_OPS:
        return MappingType.REORGANIZE
    if op in SHUFFLE_OPS:
        return MappingType.SHUFFLE
    if op in REDUCTION_OPS:
        return MappingType.REDUCTION
    if op in OPAQUE_OPS:
        return MappingType.OPAQUE
    return MappingType.OPAQUE


def _first_input_shape(node: NodeSpec) -> list[int]:
    return [int(dim) for dim in node.attrs.get("input_shape", [])]


def _get_attr_list(attrs: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = attrs.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return default


def estimate_flops(node: NodeSpec) -> float:
    if node.flops is not None:
        return float(node.flops)
    op = normalize_op(node.op_type)
    output_elems = num_elements(node.output_shape)
    if op in {"add", "sub", "mul", "div", "relu", "sigmoid", "tanh", "gelu", "clip", "batch_norm", "bias_add"}:
        flops_per_elem = {"gelu": 6.0, "sigmoid": 4.0, "tanh": 4.0}.get(op, 1.0)
        return output_elems * flops_per_elem
    if op in {"expand", "resize", "upsample", "gather", "broadcast_to"}:
        return output_elems
    if op in {"reshape", "flatten", "squeeze", "unsqueeze"}:
        return output_elems * 0.1
    if op in {"transpose", "permute", "depth_to_space", "space_to_depth"}:
        return output_elems * 0.2
    if op in {"reduce_sum", "reduce_mean", "reduce_max", "reduce_min"}:
        input_shape = _first_input_shape(node) or node.output_shape
        return float(num_elements(input_shape))
    if op == "layer_norm":
        input_shape = _first_input_shape(node) or node.output_shape
        return float(num_elements(input_shape) * 5.0)
    if op in {"avg_pool2d", "max_pool2d"}:
        kernel_h, kernel_w = _get_attr_list(node.attrs, "kernel", (2, 2))
        return float(output_elems * kernel_h * kernel_w)
    if op in {"conv1d", "conv2d", "conv3d", "depthwise_conv2d"}:
        groups = int(node.attrs.get("groups", 1))
        out_channels = int(node.attrs.get("out_channels", node.output_shape[1] if len(node.output_shape) > 1 else 1))
        in_channels = int(node.attrs.get("in_channels", out_channels))
        kernel_h, kernel_w = _get_attr_list(node.attrs, "kernel", (3, 3))
        batch = int(node.output_shape[0]) if node.output_shape else 1
        spatial = 1
        for dim in node.output_shape[2:]:
            spatial *= int(dim)
        effective_in = max(1, in_channels // max(groups, 1))
        return float(2 * batch * spatial * out_channels * effective_in * kernel_h * kernel_w)
    if op in {"gemm", "matmul"}:
        m = int(node.attrs.get("m", node.output_shape[0] if len(node.output_shape) > 0 else 1))
        n = int(node.attrs.get("n", node.output_shape[-1] if node.output_shape else 1))
        k = int(node.attrs.get("k", node.attrs.get("in_features", n)))
        return float(2 * m * n * k)
    if op in OPAQUE_OPS:
        return float(output_elems * 8.0)
    return float(output_elems)


def estimate_weight_bytes(node: NodeSpec) -> int:
    if node.weight_bytes is not None:
        return int(node.weight_bytes)
    op = normalize_op(node.op_type)
    dtype_size = dtype_nbytes(node.dtype)
    if op in {"conv1d", "conv2d", "conv3d", "depthwise_conv2d"}:
        groups = int(node.attrs.get("groups", 1))
        out_channels = int(node.attrs.get("out_channels", node.output_shape[1] if len(node.output_shape) > 1 else 1))
        in_channels = int(node.attrs.get("in_channels", out_channels))
        kernel_h, kernel_w = _get_attr_list(node.attrs, "kernel", (3, 3))
        weight_elems = out_channels * max(1, in_channels // max(groups, 1)) * kernel_h * kernel_w
        return int(weight_elems * dtype_size)
    if op in {"gemm", "matmul"}:
        k = int(node.attrs.get("k", node.attrs.get("in_features", 1)))
        n = int(node.attrs.get("n", node.output_shape[-1] if node.output_shape else 1))
        return int(k * n * dtype_size)
    if op in {"batch_norm", "layer_norm"}:
        channels = int(node.attrs.get("channels", node.output_shape[1] if len(node.output_shape) > 1 else node.output_shape[-1]))
        return int(channels * dtype_size * 2)
    if op in {"bias_add"}:
        channels = int(node.attrs.get("channels", node.output_shape[1] if len(node.output_shape) > 1 else node.output_shape[-1]))
        return int(channels * dtype_size)
    return 0


def estimate_registers_per_thread(node: NodeSpec) -> int:
    if node.registers_per_thread is not None:
        return int(node.registers_per_thread)
    op = normalize_op(node.op_type)
    input_count = max(1, len(node.inputs))
    if op in {"conv1d", "conv2d", "conv3d", "depthwise_conv2d"}:
        kernel_h, kernel_w = _get_attr_list(node.attrs, "kernel", (3, 3))
        return 32 + min(kernel_h * kernel_w, 16) + input_count
    if op in {"gemm", "matmul"}:
        return 40 + input_count
    if op in {"avg_pool2d", "max_pool2d"}:
        return 24 + input_count
    if op in REDUCTION_OPS:
        return 24 + input_count
    if op in ONE_TO_MANY_OPS:
        return 18 + input_count
    if op in SHUFFLE_OPS:
        return 14 + input_count
    if op in REORGANIZE_OPS:
        return 10 + input_count
    if op in OPAQUE_OPS:
        return 48 + input_count * 2
    return 12 + input_count


def estimate_instruction_count(node: NodeSpec) -> int:
    if node.instruction_count is not None:
        return int(node.instruction_count)
    op = normalize_op(node.op_type)
    if op in {"conv1d", "conv2d", "conv3d", "depthwise_conv2d"}:
        return 160
    if op in {"gemm", "matmul"}:
        return 180
    if op in {"avg_pool2d", "max_pool2d"}:
        return 96
    if op in REDUCTION_OPS:
        return 88
    if op in ONE_TO_MANY_OPS:
        return 48
    if op in SHUFFLE_OPS:
        return 40
    if op in REORGANIZE_OPS:
        return 24
    if op in OPAQUE_OPS:
        return 220
    return 32


def estimate_preferred_threads(node: NodeSpec) -> int:
    if node.preferred_threads is not None:
        return int(node.preferred_threads)
    op = normalize_op(node.op_type)
    if op in {"conv1d", "conv2d", "conv3d", "depthwise_conv2d", "avg_pool2d", "max_pool2d"}:
        return 128
    if op in {"gemm", "matmul", "layer_norm"}:
        return 128
    if op in REDUCTION_OPS:
        return 128
    if op in OPAQUE_OPS:
        return 128
    return 256


def estimate_tile_bytes(node: NodeSpec, threads_per_block: int) -> int:
    pattern = infer_mapping_type(node)
    multiplier = {
        MappingType.ONE_TO_ONE: 1,
        MappingType.REORGANIZE: 1,
        MappingType.SHUFFLE: 2,
        MappingType.ONE_TO_MANY: 2,
        MappingType.REDUCTION: 4,
        MappingType.MANY_TO_MANY: 8,
        MappingType.OPAQUE: 8,
    }[pattern]
    tile_elements = min(num_elements(node.output_shape), max(threads_per_block, 1) * multiplier)
    return tile_elements * dtype_nbytes(node.dtype)