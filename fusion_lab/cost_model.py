"""Hardware-aware cost model.

This file estimates block latency, occupancy and resource pressure. It is part of the reusable core and should stay experiment-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

from .enums import FusionDecision
from .fusion_rules import dominant_mapping_type, edge_fusion_decision
from .graph import GraphModel
from .hardware import HardwareProfile
from .ops import (
    estimate_flops,
    estimate_instruction_count,
    estimate_preferred_threads,
    estimate_registers_per_thread,
    estimate_tile_bytes,
    estimate_weight_bytes,
    infer_mapping_type,
)


def _soft_violation(ratio: float) -> float:
    return max(0.0, ratio - 1.0) ** 2


@dataclass
class GroupEvaluation:
    nodes: list[str]
    dominant_pattern: str
    feasible: bool
    total_latency_ms: float
    roofline_latency_ms: float
    penalty_multiplier: float
    flops: float
    memory_bytes: int
    registers_per_thread: int
    shared_mem_bytes: int
    threads_per_block: int
    occupancy: float
    instruction_count: int
    external_input_bytes: int
    external_output_bytes: int
    weight_bytes: int
    eliminated_internal_bytes: int
    fusion_hints: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "dominant_pattern": self.dominant_pattern,
            "feasible": self.feasible,
            "total_latency_ms": self.total_latency_ms,
            "roofline_latency_ms": self.roofline_latency_ms,
            "penalty_multiplier": self.penalty_multiplier,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "registers_per_thread": self.registers_per_thread,
            "shared_mem_bytes": self.shared_mem_bytes,
            "threads_per_block": self.threads_per_block,
            "occupancy": self.occupancy,
            "instruction_count": self.instruction_count,
            "external_input_bytes": self.external_input_bytes,
            "external_output_bytes": self.external_output_bytes,
            "weight_bytes": self.weight_bytes,
            "eliminated_internal_bytes": self.eliminated_internal_bytes,
            "fusion_hints": self.fusion_hints,
            "reasons": self.reasons,
        }


class CostModel:
    def __init__(
        self,
        hardware: HardwareProfile,
        enable_soft_penalties: bool = True,
        search_supported_threads: bool = True,
        penalty_weight_overrides: dict[str, float] | None = None,
    ) -> None:
        self.hardware = hardware
        self.enable_soft_penalties = enable_soft_penalties
        self.search_supported_threads = search_supported_threads
        overrides = dict(penalty_weight_overrides or {})
        unknown_keys = sorted(set(overrides) - set(self.hardware.penalty_weights))
        if unknown_keys:
            names = ", ".join(unknown_keys)
            raise ValueError(f"Unknown penalty weight override(s): {names}")
        self.penalty_weights = {**self.hardware.penalty_weights, **overrides}

    def evaluate_group(self, graph: GraphModel, node_names: Iterable[str]) -> GroupEvaluation:
        ordered_nodes = sorted(set(node_names), key=graph.topo_index)
        if not ordered_nodes:
            raise ValueError("Cannot evaluate an empty fusion group.")

        patterns = [infer_mapping_type(graph.nodes[name]) for name in ordered_nodes]
        dominant = dominant_mapping_type(patterns)
        reasons: list[str] = []
        hints: list[str] = []

        if len(ordered_nodes) > 1:
            for src, dst in graph.internal_edges(ordered_nodes):
                decision = edge_fusion_decision(
                    infer_mapping_type(graph.nodes[src]),
                    infer_mapping_type(graph.nodes[dst]),
                )
                hints.append(f"{src}->{dst}:{decision.value}")
                if decision == FusionDecision.NEVER:
                    reasons.append(f"illegal pairwise fusion on edge {src}->{dst}")

        if reasons:
            return GroupEvaluation(
                nodes=ordered_nodes,
                dominant_pattern=dominant.value,
                feasible=False,
                total_latency_ms=float("inf"),
                roofline_latency_ms=float("inf"),
                penalty_multiplier=float("inf"),
                flops=0.0,
                memory_bytes=0,
                registers_per_thread=0,
                shared_mem_bytes=0,
                threads_per_block=0,
                occupancy=0.0,
                instruction_count=0,
                external_input_bytes=0,
                external_output_bytes=0,
                weight_bytes=0,
                eliminated_internal_bytes=0,
                fusion_hints=hints,
                reasons=reasons,
            )

        preferred_threads = [estimate_preferred_threads(graph.nodes[name]) for name in ordered_nodes]
        node_work_weights = [max(estimate_flops(graph.nodes[name]), 1.0) for name in ordered_nodes]
        flops = sum(node_work_weights)
        weight_bytes = sum(estimate_weight_bytes(graph.nodes[name]) for name in ordered_nodes)
        instruction_count = sum(estimate_instruction_count(graph.nodes[name]) for name in ordered_nodes)

        node_regs = [estimate_registers_per_thread(graph.nodes[name]) for name in ordered_nodes]
        registers_per_thread = max(node_regs)
        if len(node_regs) > 1:
            registers_per_thread += sum(int(value * 0.35) for value in node_regs[1:])
            registers_per_thread += 2 * (len(node_regs) - 1)

        external_inputs = graph.external_inputs_for_group(ordered_nodes)
        external_outputs = graph.external_output_nodes_for_group(ordered_nodes)
        external_input_bytes = sum(graph.tensor_bytes(ref) for ref in external_inputs)
        external_output_bytes = sum(graph.nodes[name].output_bytes for name in external_outputs)

        internal_only_outputs = set(ordered_nodes) - set(external_outputs)
        eliminated_internal_bytes = sum(graph.nodes[name].output_bytes for name in internal_only_outputs)
        memory_bytes = external_input_bytes + external_output_bytes + weight_bytes

        if self.search_supported_threads:
            candidate_threads = [
                threads
                for threads in self.hardware.supported_threads
                if threads <= self.hardware.max_threads_per_block
            ]
            if not candidate_threads:
                candidate_threads = [min(min(preferred_threads), self.hardware.max_threads_per_block)]
        else:
            candidate_threads = [min(min(preferred_threads), self.hardware.max_threads_per_block)]

        evaluations = [
            self._evaluate_candidate(
                graph=graph,
                ordered_nodes=ordered_nodes,
                dominant_pattern=dominant.value,
                preferred_threads=preferred_threads,
                node_work_weights=node_work_weights,
                chosen_threads=chosen_threads,
                flops=flops,
                memory_bytes=memory_bytes,
                registers_per_thread=registers_per_thread,
                instruction_count=instruction_count,
                external_input_bytes=external_input_bytes,
                external_output_bytes=external_output_bytes,
                weight_bytes=weight_bytes,
                eliminated_internal_bytes=eliminated_internal_bytes,
                fusion_hints=hints,
            )
            for chosen_threads in candidate_threads
        ]

        feasible_evaluations = [evaluation for evaluation in evaluations if evaluation.feasible]
        if feasible_evaluations:
            return min(feasible_evaluations, key=lambda item: item.total_latency_ms)

        best_failed = min(
            evaluations,
            key=lambda item: (
                len(item.reasons),
                item.shared_mem_bytes if item.shared_mem_bytes > 0 else float("inf"),
                item.threads_per_block if item.threads_per_block > 0 else float("inf"),
            ),
        )
        merged_reasons: list[str] = []
        seen_reasons: set[str] = set()
        for evaluation in evaluations:
            for reason in evaluation.reasons:
                if reason not in seen_reasons:
                    seen_reasons.add(reason)
                    merged_reasons.append(reason)
        best_failed.reasons = merged_reasons or best_failed.reasons
        return best_failed

    def _evaluate_candidate(
        self,
        graph: GraphModel,
        ordered_nodes: list[str],
        dominant_pattern: str,
        preferred_threads: list[int],
        node_work_weights: list[float],
        chosen_threads: int,
        flops: float,
        memory_bytes: int,
        registers_per_thread: int,
        instruction_count: int,
        external_input_bytes: int,
        external_output_bytes: int,
        weight_bytes: int,
        eliminated_internal_bytes: int,
        fusion_hints: list[str],
    ) -> GroupEvaluation:
        reasons: list[str] = []
        shared_mem_bytes = self._estimate_peak_live_tile_bytes(graph, ordered_nodes, chosen_threads)
        group_regs_per_block = registers_per_thread * chosen_threads

        if registers_per_thread > self.hardware.registers_per_thread_limit:
            reasons.append("registers per thread exceed hard limit")
        if group_regs_per_block > self.hardware.registers_per_sm:
            reasons.append("register file cannot host one block")
        if shared_mem_bytes > self.hardware.shared_mem_per_block:
            reasons.append("shared memory per block exceeds hard limit")
        if shared_mem_bytes > self.hardware.shared_mem_per_sm:
            reasons.append("shared memory per SM cannot host one block")
        if chosen_threads > self.hardware.max_threads_per_block:
            reasons.append("threads per block exceed hard limit")

        occupancy = 0.0
        if not reasons:
            blocks_by_regs = self.hardware.max_blocks_per_sm if group_regs_per_block == 0 else self.hardware.registers_per_sm // group_regs_per_block
            blocks_by_smem = self.hardware.max_blocks_per_sm if shared_mem_bytes == 0 else self.hardware.shared_mem_per_sm // shared_mem_bytes
            blocks_by_threads = self.hardware.max_threads_per_sm // chosen_threads
            active_blocks = min(
                self.hardware.max_blocks_per_sm,
                blocks_by_regs,
                blocks_by_smem,
                blocks_by_threads,
            )
            if active_blocks < 1:
                reasons.append("occupancy falls to zero")
            else:
                active_warps = active_blocks * ceil(chosen_threads / self.hardware.warp_size)
                occupancy = min(1.0, active_warps / self.hardware.max_warps_per_sm)

        if reasons:
            return GroupEvaluation(
                nodes=ordered_nodes,
                dominant_pattern=dominant_pattern,
                feasible=False,
                total_latency_ms=float("inf"),
                roofline_latency_ms=float("inf"),
                penalty_multiplier=float("inf"),
                flops=flops,
                memory_bytes=memory_bytes,
                registers_per_thread=registers_per_thread,
                shared_mem_bytes=shared_mem_bytes,
                threads_per_block=chosen_threads,
                occupancy=occupancy,
                instruction_count=instruction_count,
                external_input_bytes=external_input_bytes,
                external_output_bytes=external_output_bytes,
                weight_bytes=weight_bytes,
                eliminated_internal_bytes=eliminated_internal_bytes,
                fusion_hints=list(fusion_hints),
                reasons=reasons,
            )

        compute_eff = self.hardware.base_compute_efficiency[dominant_pattern] * max(occupancy, 0.25)
        memory_eff = self.hardware.base_memory_efficiency[dominant_pattern] * (0.5 + 0.5 * occupancy)
        roofline_s = max(
            flops / max(self.hardware.peak_flops * compute_eff, 1e-9),
            memory_bytes / max(self.hardware.memory_bandwidth * memory_eff, 1e-9),
        ) + self.hardware.launch_overhead_s

        penalty_multiplier = 1.0
        if self.enable_soft_penalties:
            reg_ratio = registers_per_thread / max(
                self.hardware.registers_per_thread_limit * self.hardware.reg_soft_ratio,
                1e-9,
            )
            smem_ratio = shared_mem_bytes / max(
                self.hardware.shared_mem_per_block * self.hardware.smem_soft_ratio,
                1e-9,
            )
            geometry_penalty = self._geometry_penalty(preferred_threads, chosen_threads, node_work_weights)
            icache_ratio = instruction_count / max(self.hardware.icache_inst_limit, 1e-9)
            weights = self.penalty_weights
            penalty_multiplier += (
                weights["register"] * _soft_violation(reg_ratio)
                + weights["shared_memory"] * _soft_violation(smem_ratio)
                + weights["geometry"] * geometry_penalty
                + weights["icache"] * _soft_violation(icache_ratio)
            )

        total_latency_ms = roofline_s * penalty_multiplier * 1e3
        return GroupEvaluation(
            nodes=ordered_nodes,
            dominant_pattern=dominant_pattern,
            feasible=True,
            total_latency_ms=total_latency_ms,
            roofline_latency_ms=roofline_s * 1e3,
            penalty_multiplier=penalty_multiplier,
            flops=flops,
            memory_bytes=memory_bytes,
            registers_per_thread=registers_per_thread,
            shared_mem_bytes=shared_mem_bytes,
            threads_per_block=chosen_threads,
            occupancy=occupancy,
            instruction_count=instruction_count,
            external_input_bytes=external_input_bytes,
            external_output_bytes=external_output_bytes,
            weight_bytes=weight_bytes,
            eliminated_internal_bytes=eliminated_internal_bytes,
            fusion_hints=list(fusion_hints),
            reasons=[],
        )

    @staticmethod
    def _geometry_penalty(
        preferred_threads: list[int],
        chosen_threads: int,
        node_work_weights: list[float] | None = None,
    ) -> float:
        if not preferred_threads or chosen_threads <= 0:
            return 0.0
        weights = node_work_weights or [1.0] * len(preferred_threads)
        total_weight = sum(max(weight, 1e-9) for weight in weights)
        penalty = 0.0
        for preferred, weight in zip(preferred_threads, weights):
            high = max(preferred, chosen_threads)
            low = max(min(preferred, chosen_threads), 1)
            penalty += ((high - low) / high) * max(weight, 1e-9)
        return penalty / max(total_weight, 1e-9)

    @staticmethod
    def _estimate_peak_live_tile_bytes(
        graph: GraphModel,
        ordered_nodes: list[str],
        chosen_threads: int,
    ) -> int:
        group = set(ordered_nodes)
        remaining_internal_uses = {
            name: sum(1 for succ in graph.successors(name) if succ in group)
            for name in ordered_nodes
        }
        live_tiles: dict[str, int] = {}
        peak = 0
        for node_name in ordered_nodes:
            node = graph.nodes[node_name]
            if remaining_internal_uses[node_name] > 0:
                live_tiles[node_name] = estimate_tile_bytes(node, chosen_threads)
            peak = max(peak, sum(live_tiles.values()))
            for pred in graph.predecessors(node_name):
                if pred in group:
                    remaining_internal_uses[pred] -= 1
                    if remaining_internal_uses[pred] <= 0:
                        live_tiles.pop(pred, None)
            peak = max(peak, sum(live_tiles.values()))
        return peak
