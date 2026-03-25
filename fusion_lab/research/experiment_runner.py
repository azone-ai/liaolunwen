"""Experiment-only runner and report generation.

This file is for study outputs such as summaries, CSV/Markdown reports and comparison plots. It should not contain reusable search logic.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

from ..cost_model import CostModel
from ..graph import GraphModel
from ..hardware import HardwareProfile
from ..search import PlanResult, greedy_merge_layers, optimize_layers_dp, single_layer_plan


DEFAULT_METHODS = ("none", "greedy", "dp_paper", "hw_aware")
ALL_METHODS = DEFAULT_METHODS + ("hw_no_thread_search",)


def run_methods(
    graph: GraphModel,
    hardware: HardwareProfile,
    methods: Iterable[str] = DEFAULT_METHODS,
    max_depth: int = 4,
) -> list[PlanResult]:
    results: list[PlanResult] = []
    for method in methods:
        start_time = perf_counter()
        if method == "none":
            result = single_layer_plan(
                graph,
                CostModel(hardware, enable_soft_penalties=False),
                method="none",
                hardware_name=hardware.name,
            )
        elif method == "greedy":
            result = greedy_merge_layers(
                graph,
                CostModel(hardware, enable_soft_penalties=False),
                max_depth=max_depth,
                method="greedy",
                hardware_name=hardware.name,
            )
        elif method == "dp_paper":
            result = optimize_layers_dp(
                graph,
                CostModel(hardware, enable_soft_penalties=False),
                max_depth=max_depth,
                method="dp_paper",
                hardware_name=hardware.name,
            )
        elif method == "hw_aware":
            result = optimize_layers_dp(
                graph,
                CostModel(hardware, enable_soft_penalties=True),
                max_depth=max_depth,
                method="hw_aware",
                hardware_name=hardware.name,
            )
        elif method == "hw_no_thread_search":
            result = optimize_layers_dp(
                graph,
                CostModel(hardware, enable_soft_penalties=True, search_supported_threads=False),
                max_depth=max_depth,
                method="hw_no_thread_search",
                hardware_name=hardware.name,
            )
        else:
            raise ValueError(f"Unsupported method '{method}'.")
        result.search_time_ms = (perf_counter() - start_time) * 1000.0
        results.append(result)
    return results


def write_reports(results: list[PlanResult], out_dir: str | Path, with_plots: bool = False) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    baseline = next((item for item in results if item.method == "none"), None)
    baseline_latency = baseline.estimated_latency_ms if baseline and baseline.feasible else None
    baseline_kernels = baseline.kernel_count if baseline and baseline.feasible else None
    baseline_memory = baseline.total_memory_bytes if baseline and baseline.feasible else None

    for result in results:
        plan_path = output_dir / f"{result.method}_plan.json"
        with open(plan_path, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2, ensure_ascii=False)

        speedup = None
        if baseline_latency and result.feasible and result.estimated_latency_ms > 0:
            speedup = baseline_latency / result.estimated_latency_ms

        kernel_reduction = None
        if baseline_kernels and baseline_kernels > 0 and result.feasible:
            kernel_reduction = 1.0 - (result.kernel_count / baseline_kernels)

        memory_reduction = None
        if baseline_memory and baseline_memory > 0 and result.feasible:
            memory_reduction = 1.0 - (result.total_memory_bytes / baseline_memory)

        summary_rows.append(
            {
                "method": result.method,
                "graph_name": result.graph_name,
                "hardware_name": result.hardware_name,
                "feasible": result.feasible,
                "estimated_latency_ms": result.estimated_latency_ms,
                "kernel_count": result.kernel_count,
                "speedup_vs_none": speedup,
                "kernel_reduction_vs_none": kernel_reduction,
                "average_ops_per_kernel": result.average_ops_per_kernel,
                "total_nodes": result.total_nodes,
                "total_flops": result.total_flops,
                "total_memory_bytes": result.total_memory_bytes,
                "total_external_input_bytes": result.total_external_input_bytes,
                "total_external_output_bytes": result.total_external_output_bytes,
                "total_weight_bytes": result.total_weight_bytes,
                "total_eliminated_internal_bytes": result.total_eliminated_internal_bytes,
                "memory_reduction_vs_none": memory_reduction,
                "avg_occupancy": result.avg_occupancy,
                "avg_penalty_multiplier": result.avg_penalty_multiplier,
                "avg_registers_per_thread": result.avg_registers_per_thread,
                "max_registers_per_thread": result.max_registers_per_thread,
                "avg_shared_mem_bytes": result.avg_shared_mem_bytes,
                "max_shared_mem_bytes": result.max_shared_mem_bytes,
                "avg_threads_per_block": result.avg_threads_per_block,
                "max_threads_per_block": result.max_threads_per_block,
                "search_time_ms": result.search_time_ms,
            }
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, ensure_ascii=False)

    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_lines = [
        f"# {results[0].graph_name} on {results[0].hardware_name}",
        "",
        "| Method | Feasible | Latency (ms) | Search Time (ms) | Speedup | Kernels | Kernel Red. | Memory (MiB) | Saved Internal (MiB) | Avg Occ. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        speedup = "-" if row["speedup_vs_none"] is None else f"{row['speedup_vs_none']:.3f}"
        kernel_reduction = "-" if row["kernel_reduction_vs_none"] is None else f"{row['kernel_reduction_vs_none'] * 100:.1f}%"
        latency = "inf" if row["estimated_latency_ms"] == float("inf") else f"{row['estimated_latency_ms']:.4f}"
        memory_mib = _bytes_to_mib(row["total_memory_bytes"])
        saved_mib = _bytes_to_mib(row["total_eliminated_internal_bytes"])
        markdown_lines.append(
            f"| {row['method']} | {row['feasible']} | {latency} | {row['search_time_ms']:.3f} | {speedup} | {row['kernel_count']} | {kernel_reduction} | {memory_mib:.3f} | {saved_mib:.3f} | {row['avg_occupancy']:.3f} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Search Time (ms)` is the measured wall-clock search and plan-construction time for one method on the current machine.",
            "- `Memory (MiB)` is the estimated external traffic used by the cost model: inputs + outputs + weights.",
            "- `Saved Internal (MiB)` is the estimated intermediate output traffic eliminated by fusion.",
            "- `Kernel Red.` is measured against the `none` baseline.",
        ]
    )
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines))

    hardware_rows = [
        {
            "method": row["method"],
            "feasible": row["feasible"],
            "avg_occupancy": row["avg_occupancy"],
            "avg_registers_per_thread": row["avg_registers_per_thread"],
            "max_registers_per_thread": row["max_registers_per_thread"],
            "avg_shared_mem_kib": row["avg_shared_mem_bytes"] / 1024.0,
            "max_shared_mem_kib": row["max_shared_mem_bytes"] / 1024.0,
            "avg_threads_per_block": row["avg_threads_per_block"],
            "max_threads_per_block": row["max_threads_per_block"],
            "avg_penalty_multiplier": row["avg_penalty_multiplier"],
        }
        for row in summary_rows
    ]
    with open(output_dir / "hardware_analysis.json", "w", encoding="utf-8") as handle:
        json.dump(hardware_rows, handle, indent=2, ensure_ascii=False)

    hardware_markdown_lines = [
        f"# Hardware Analysis: {results[0].graph_name} on {results[0].hardware_name}",
        "",
        "| Method | Feasible | Avg Occ. | Avg Reg/Thr | Max Reg/Thr | Avg SMem/Block (KiB) | Max SMem/Block (KiB) | Avg Threads/Block | Max Threads/Block | Avg Penalty |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in hardware_rows:
        hardware_markdown_lines.append(
            f"| {row['method']} | {row['feasible']} | {row['avg_occupancy']:.3f} | {row['avg_registers_per_thread']:.2f} | {row['max_registers_per_thread']} | {row['avg_shared_mem_kib']:.3f} | {row['max_shared_mem_kib']:.3f} | {row['avg_threads_per_block']:.2f} | {row['max_threads_per_block']} | {row['avg_penalty_multiplier']:.4f} |"
        )
    hardware_markdown_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Avg` metrics are latency-weighted averages across fused blocks in the current plan.",
            "- Shared memory metrics are reported in KiB per block.",
            "- `Avg Penalty` reflects the average soft-penalty multiplier under the current cost model.",
        ]
    )
    with open(output_dir / "hardware_analysis.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(hardware_markdown_lines))

    if with_plots:
        _write_plot(summary_rows, output_dir)


def _bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _write_plot(summary_rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    methods = [row["method"] for row in summary_rows]
    latencies = [
        0.0 if row["estimated_latency_ms"] == float("inf") else row["estimated_latency_ms"]
        for row in summary_rows
    ]
    plt.figure(figsize=(8, 4))
    bars = plt.bar(methods, latencies, color=["#7e9fbe", "#7bb274", "#e0a458", "#d35d6e"])
    plt.ylabel("Estimated Latency (ms)")
    plt.title(f"{summary_rows[0]['graph_name']} on {summary_rows[0]['hardware_name']}")
    for bar, latency in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width() / 2, latency, f"{latency:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=160)
    plt.close()
