"""Penalty calibration helpers for hardware-aware fusion experiments.

This module keeps penalty-profile tuning out of the reusable cost model. It
provides named calibration profiles plus lightweight graph-feature heuristics
for choosing a profile automatically during experiments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..graph import GraphModel
from ..hardware import HardwareProfile
from ..ops import normalize_op


BASELINE_PROFILE = "baseline"
GLOBAL_SOFT_PROFILE = "global_soft"
AUTO_PROFILE = "auto"
CALIBRATION_PROFILES = (
    BASELINE_PROFILE,
    GLOBAL_SOFT_PROFILE,
    "cnn_seq",
    "cnn_residual",
    "cnn_light",
    "reduction_tf",
)
PROFILE_CHOICES = CALIBRATION_PROFILES + (AUTO_PROFILE,)


PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    GLOBAL_SOFT_PROFILE: {
        "penalty_weights": {
            "register": 0.18,
            "shared_memory": 0.12,
            "geometry": 0.10,
            "icache": 0.06,
        },
        "reg_soft_ratio": 0.60,
        "smem_soft_ratio": 0.65,
    },
    "cnn_seq": {
        "penalty_weights": {
            "register": 0.10,
            "shared_memory": 0.08,
            "geometry": 0.08,
            "icache": 0.05,
        },
        "reg_soft_ratio": 0.65,
        "smem_soft_ratio": 0.70,
    },
    "cnn_residual": {
        "penalty_weights": {
            "register": 0.18,
            "shared_memory": 0.12,
            "geometry": 0.12,
            "icache": 0.08,
        },
        "reg_soft_ratio": 0.60,
        "smem_soft_ratio": 0.65,
    },
    "cnn_light": {
        "penalty_weights": {
            "register": 0.12,
            "shared_memory": 0.08,
            "geometry": 0.08,
            "icache": 0.05,
        },
        "reg_soft_ratio": 0.65,
        "smem_soft_ratio": 0.70,
    },
    "reduction_tf": {
        "penalty_weights": {
            "register": 0.25,
            "shared_memory": 0.15,
            "geometry": 0.08,
            "icache": 0.15,
        },
        "reg_soft_ratio": 0.55,
        "smem_soft_ratio": 0.60,
    },
}


@dataclass(frozen=True)
class GraphFeatureSummary:
    total_nodes: int
    depthwise_ratio: float
    reduction_ratio: float
    residual_add_ratio: float
    sequential_conv_ratio: float
    op_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "depthwise_ratio": self.depthwise_ratio,
            "reduction_ratio": self.reduction_ratio,
            "residual_add_ratio": self.residual_add_ratio,
            "sequential_conv_ratio": self.sequential_conv_ratio,
            "op_counts": dict(self.op_counts),
        }


def summarize_graph_features(graph: GraphModel) -> GraphFeatureSummary:
    total_nodes = max(len(graph.nodes), 1)
    op_counts: Counter[str] = Counter()
    depthwise_count = 0
    reduction_count = 0
    residual_add_count = 0
    conv_like_count = 0

    reduction_ops = {"reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "layer_norm", "softmax"}
    conv_ops = {"conv1d", "conv2d", "conv3d", "depthwise_conv2d"}

    for node in graph.nodes.values():
        op = normalize_op(node.op_type)
        op_counts[op] += 1

        if op in reduction_ops:
            reduction_count += 1

        if op == "depthwise_conv2d":
            depthwise_count += 1
        elif op == "conv2d":
            groups = int(node.attrs.get("group", node.attrs.get("groups", 1)))
            in_channels = int(node.attrs.get("in_channels", 1))
            out_channels = int(node.attrs.get("out_channels", 1))
            if groups > 1 and groups == in_channels and out_channels % max(groups, 1) == 0:
                depthwise_count += 1

        if op in conv_ops:
            conv_like_count += 1

        if op == "add":
            graph_inputs = [ref for ref in node.inputs if ref in graph.nodes]
            external_inputs = [ref for ref in node.inputs if ref in graph.inputs]
            if len(node.inputs) >= 2:
                if graph_inputs and external_inputs:
                    residual_add_count += 1
                elif len(graph_inputs) >= 2:
                    pred_layers = [graph.layer_index(ref) for ref in graph_inputs]
                    if pred_layers and (max(pred_layers) - min(pred_layers) >= 1):
                        residual_add_count += 1

    return GraphFeatureSummary(
        total_nodes=total_nodes,
        depthwise_ratio=depthwise_count / total_nodes,
        reduction_ratio=reduction_count / total_nodes,
        residual_add_ratio=residual_add_count / total_nodes,
        sequential_conv_ratio=conv_like_count / total_nodes,
        op_counts=dict(op_counts),
    )


def choose_profile_for_graph(graph: GraphModel) -> tuple[str, GraphFeatureSummary]:
    summary = summarize_graph_features(graph)
    if summary.depthwise_ratio >= 0.08:
        return "cnn_light", summary
    if summary.reduction_ratio >= 0.08:
        return "reduction_tf", summary
    if summary.residual_add_ratio >= 0.03:
        return "cnn_residual", summary
    return "cnn_seq", summary


def resolve_profile_name(graph: GraphModel, requested_profile: str | None) -> tuple[str, GraphFeatureSummary]:
    profile = requested_profile or BASELINE_PROFILE
    if profile == AUTO_PROFILE:
        return choose_profile_for_graph(graph)
    if profile not in CALIBRATION_PROFILES:
        raise ValueError(f"Unsupported penalty calibration profile '{profile}'.")
    return profile, summarize_graph_features(graph)


def apply_penalty_profile(
    hardware: HardwareProfile,
    graph: GraphModel,
    requested_profile: str | None = None,
) -> tuple[HardwareProfile, str, GraphFeatureSummary]:
    profile_name, summary = resolve_profile_name(graph, requested_profile)
    if profile_name == BASELINE_PROFILE:
        return hardware, profile_name, summary

    overrides = PROFILE_OVERRIDES[profile_name]
    calibrated = hardware.with_overrides(
        name=f"{hardware.name}:{profile_name}",
        **overrides,
    )
    return calibrated, profile_name, summary
