"""Fusion search algorithms.

This file contains single-op baselines, greedy fusion and DP-based fusion search, plus plan/result structures used by reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cost_model import CostModel, GroupEvaluation
from .graph import GraphModel


@dataclass
class PlanBlock:
    block_id: int
    layer_start: int
    layer_end: int
    nodes: list[str]
    cost_ms: float
    roofline_latency_ms: float
    penalty_multiplier: float
    dominant_pattern: str
    flops: float
    memory_bytes: int
    occupancy: float
    registers_per_thread: int
    shared_mem_bytes: int
    threads_per_block: int
    instruction_count: int
    external_input_bytes: int
    external_output_bytes: int
    weight_bytes: int
    eliminated_internal_bytes: int
    fusion_hints: list[str]

    @classmethod
    def from_evaluation(
        cls,
        block_id: int,
        layer_start: int,
        layer_end: int,
        evaluation: GroupEvaluation,
    ) -> "PlanBlock":
        return cls(
            block_id=block_id,
            layer_start=layer_start,
            layer_end=layer_end,
            nodes=evaluation.nodes,
            cost_ms=evaluation.total_latency_ms,
            roofline_latency_ms=evaluation.roofline_latency_ms,
            penalty_multiplier=evaluation.penalty_multiplier,
            dominant_pattern=evaluation.dominant_pattern,
            flops=evaluation.flops,
            memory_bytes=evaluation.memory_bytes,
            occupancy=evaluation.occupancy,
            registers_per_thread=evaluation.registers_per_thread,
            shared_mem_bytes=evaluation.shared_mem_bytes,
            threads_per_block=evaluation.threads_per_block,
            instruction_count=evaluation.instruction_count,
            external_input_bytes=evaluation.external_input_bytes,
            external_output_bytes=evaluation.external_output_bytes,
            weight_bytes=evaluation.weight_bytes,
            eliminated_internal_bytes=evaluation.eliminated_internal_bytes,
            fusion_hints=evaluation.fusion_hints,
        )

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "layer_start": self.layer_start,
            "layer_end": self.layer_end,
            "nodes": self.nodes,
            "cost_ms": self.cost_ms,
            "roofline_latency_ms": self.roofline_latency_ms,
            "penalty_multiplier": self.penalty_multiplier,
            "dominant_pattern": self.dominant_pattern,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "occupancy": self.occupancy,
            "registers_per_thread": self.registers_per_thread,
            "shared_mem_bytes": self.shared_mem_bytes,
            "threads_per_block": self.threads_per_block,
            "instruction_count": self.instruction_count,
            "external_input_bytes": self.external_input_bytes,
            "external_output_bytes": self.external_output_bytes,
            "weight_bytes": self.weight_bytes,
            "eliminated_internal_bytes": self.eliminated_internal_bytes,
            "fusion_hints": self.fusion_hints,
        }


@dataclass
class IntervalEvaluation:
    layer_start: int
    layer_end: int
    feasible: bool
    cost_ms: float
    evaluations: list[GroupEvaluation]
    reasons: list[str]


@dataclass
class PlanResult:
    method: str
    graph_name: str
    hardware_name: str
    estimated_latency_ms: float
    kernel_count: int
    intervals: list[tuple[int, int]]
    blocks: list[PlanBlock]
    total_nodes: int
    average_ops_per_kernel: float
    total_flops: float
    total_memory_bytes: int
    total_external_input_bytes: int
    total_external_output_bytes: int
    total_weight_bytes: int
    total_eliminated_internal_bytes: int
    avg_occupancy: float
    avg_penalty_multiplier: float
    feasible: bool = True

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "graph_name": self.graph_name,
            "hardware_name": self.hardware_name,
            "estimated_latency_ms": self.estimated_latency_ms,
            "kernel_count": self.kernel_count,
            "intervals": self.intervals,
            "blocks": [block.to_dict() for block in self.blocks],
            "total_nodes": self.total_nodes,
            "average_ops_per_kernel": self.average_ops_per_kernel,
            "total_flops": self.total_flops,
            "total_memory_bytes": self.total_memory_bytes,
            "total_external_input_bytes": self.total_external_input_bytes,
            "total_external_output_bytes": self.total_external_output_bytes,
            "total_weight_bytes": self.total_weight_bytes,
            "total_eliminated_internal_bytes": self.total_eliminated_internal_bytes,
            "avg_occupancy": self.avg_occupancy,
            "avg_penalty_multiplier": self.avg_penalty_multiplier,
            "feasible": self.feasible,
        }


def evaluate_interval(
    graph: GraphModel,
    cost_model: CostModel,
    layers: list[list[str]],
    layer_start: int,
    layer_end: int,
) -> IntervalEvaluation:
    node_set = {
        node_name
        for layer_id in range(layer_start, layer_end + 1)
        for node_name in layers[layer_id]
    }
    components = graph.weakly_connected_components(node_set)
    evaluations: list[GroupEvaluation] = []
    reasons: list[str] = []
    total_cost = 0.0
    for component in components:
        evaluation = cost_model.evaluate_group(graph, component)
        evaluations.append(evaluation)
        if not evaluation.feasible:
            reasons.extend(evaluation.reasons)
        else:
            total_cost += evaluation.total_latency_ms
    feasible = not reasons
    return IntervalEvaluation(
        layer_start=layer_start,
        layer_end=layer_end,
        feasible=feasible,
        cost_ms=total_cost if feasible else float("inf"),
        evaluations=evaluations,
        reasons=reasons,
    )


def build_plan_result(
    graph: GraphModel,
    hardware_name: str,
    method: str,
    interval_evaluations: Iterable[IntervalEvaluation],
) -> PlanResult:
    intervals: list[tuple[int, int]] = []
    blocks: list[PlanBlock] = []
    total_latency = 0.0
    block_id = 0
    feasible = True
    for interval in interval_evaluations:
        intervals.append((interval.layer_start, interval.layer_end))
        feasible = feasible and interval.feasible
        if not interval.feasible:
            continue
        for evaluation in interval.evaluations:
            if evaluation.feasible:
                blocks.append(
                    PlanBlock.from_evaluation(
                        block_id=block_id,
                        layer_start=interval.layer_start,
                        layer_end=interval.layer_end,
                        evaluation=evaluation,
                    )
                )
                total_latency += evaluation.total_latency_ms
                block_id += 1

    total_nodes = sum(len(block.nodes) for block in blocks)
    average_ops_per_kernel = total_nodes / len(blocks) if blocks else 0.0
    total_flops = sum(block.flops for block in blocks)
    total_memory_bytes = sum(block.memory_bytes for block in blocks)
    total_external_input_bytes = sum(block.external_input_bytes for block in blocks)
    total_external_output_bytes = sum(block.external_output_bytes for block in blocks)
    total_weight_bytes = sum(block.weight_bytes for block in blocks)
    total_eliminated_internal_bytes = sum(block.eliminated_internal_bytes for block in blocks)

    if total_latency > 0:
        avg_occupancy = sum(block.occupancy * block.cost_ms for block in blocks) / total_latency
        avg_penalty_multiplier = sum(block.penalty_multiplier * block.cost_ms for block in blocks) / total_latency
    elif blocks:
        avg_occupancy = sum(block.occupancy for block in blocks) / len(blocks)
        avg_penalty_multiplier = sum(block.penalty_multiplier for block in blocks) / len(blocks)
    else:
        avg_occupancy = 0.0
        avg_penalty_multiplier = 0.0

    return PlanResult(
        method=method,
        graph_name=graph.name,
        hardware_name=hardware_name,
        estimated_latency_ms=total_latency if feasible else float("inf"),
        kernel_count=len(blocks),
        intervals=intervals,
        blocks=blocks,
        total_nodes=total_nodes,
        average_ops_per_kernel=average_ops_per_kernel,
        total_flops=total_flops,
        total_memory_bytes=total_memory_bytes,
        total_external_input_bytes=total_external_input_bytes,
        total_external_output_bytes=total_external_output_bytes,
        total_weight_bytes=total_weight_bytes,
        total_eliminated_internal_bytes=total_eliminated_internal_bytes,
        avg_occupancy=avg_occupancy,
        avg_penalty_multiplier=avg_penalty_multiplier,
        feasible=feasible,
    )


def single_layer_plan(graph: GraphModel, cost_model: CostModel, method: str, hardware_name: str) -> PlanResult:
    layers = graph.layers()
    intervals = [
        evaluate_interval(graph, cost_model, layers, layer_idx, layer_idx)
        for layer_idx in range(len(layers))
    ]
    return build_plan_result(graph, hardware_name, method, intervals)


def greedy_merge_layers(
    graph: GraphModel,
    cost_model: CostModel,
    max_depth: int,
    method: str,
    hardware_name: str,
) -> PlanResult:
    layers = graph.layers()
    partitions: list[tuple[int, int]] = [(idx, idx) for idx in range(len(layers))]
    cache: dict[tuple[int, int], IntervalEvaluation] = {}

    def cached_eval(start: int, end: int) -> IntervalEvaluation:
        key = (start, end)
        if key not in cache:
            cache[key] = evaluate_interval(graph, cost_model, layers, start, end)
        return cache[key]

    improved = True
    while improved and len(partitions) > 1:
        improved = False
        best_gain = 0.0
        best_idx: int | None = None
        for idx in range(len(partitions) - 1):
            left = partitions[idx]
            right = partitions[idx + 1]
            merged = (left[0], right[1])
            if merged[1] - merged[0] + 1 > max_depth:
                continue
            merged_eval = cached_eval(*merged)
            if not merged_eval.feasible:
                continue
            separate_cost = cached_eval(*left).cost_ms + cached_eval(*right).cost_ms
            gain = separate_cost - merged_eval.cost_ms
            if gain > best_gain + 1e-9:
                best_gain = gain
                best_idx = idx
        if best_idx is not None:
            left = partitions[best_idx]
            right = partitions[best_idx + 1]
            partitions = (
                partitions[:best_idx]
                + [(left[0], right[1])]
                + partitions[best_idx + 2 :]
            )
            improved = True

    interval_evaluations = [cached_eval(start, end) for start, end in partitions]
    return build_plan_result(graph, hardware_name, method, interval_evaluations)


def optimize_layers_dp(
    graph: GraphModel,
    cost_model: CostModel,
    max_depth: int,
    method: str,
    hardware_name: str,
) -> PlanResult:
    layers = graph.layers()
    cache: dict[tuple[int, int], IntervalEvaluation] = {}

    def cached_eval(start: int, end: int) -> IntervalEvaluation:
        key = (start, end)
        if key not in cache:
            cache[key] = evaluate_interval(graph, cost_model, layers, start, end)
        return cache[key]

    layer_count = len(layers)
    dp = [float("inf")] * (layer_count + 1)
    prev: list[tuple[int, int] | None] = [None] * (layer_count + 1)
    dp[0] = 0.0

    for end in range(1, layer_count + 1):
        for start in range(max(0, end - max_depth), end):
            interval = cached_eval(start, end - 1)
            if not interval.feasible or dp[start] == float("inf"):
                continue
            candidate = dp[start] + interval.cost_ms
            if candidate < dp[end]:
                dp[end] = candidate
                prev[end] = (start, end - 1)

    if dp[layer_count] == float("inf"):
        return PlanResult(
            method=method,
            graph_name=graph.name,
            hardware_name=hardware_name,
            estimated_latency_ms=float("inf"),
            kernel_count=0,
            intervals=[],
            blocks=[],
            total_nodes=0,
            average_ops_per_kernel=0.0,
            total_flops=0.0,
            total_memory_bytes=0,
            total_external_input_bytes=0,
            total_external_output_bytes=0,
            total_weight_bytes=0,
            total_eliminated_internal_bytes=0,
            avg_occupancy=0.0,
            avg_penalty_multiplier=0.0,
            feasible=False,
        )

    intervals_rev: list[tuple[int, int]] = []
    cursor = layer_count
    while cursor > 0:
        choice = prev[cursor]
        if choice is None:
            raise RuntimeError("Failed to backtrack DP plan.")
        intervals_rev.append(choice)
        cursor = choice[0]
    intervals = list(reversed(intervals_rev))
    interval_evaluations = [cached_eval(start, end) for start, end in intervals]
    return build_plan_result(graph, hardware_name, method, interval_evaluations)
