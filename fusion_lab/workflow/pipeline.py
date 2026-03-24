"""End-to-end workflow orchestration.

This file connects inputs, hardware profiles, search, report generation, ONNX export and optional runtime validation into complete workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..graph import GraphModel
from ..hardware import HardwareProfile
from ..onnx_bridge import export_fused_onnx, load_onnx_model
from ..research.experiment_runner import run_methods, write_reports
from ..research.runtime_validation import (
    benchmark_with_onnxruntime,
    generate_random_inputs,
    verify_models,
    write_benchmark_report,
    write_verification_report,
)


def run_all_sample_workflow(
    hardware_path: str | Path,
    out_dir: str | Path,
    methods: Iterable[str],
    max_depth: int,
    plots: bool,
) -> None:
    hardware = HardwareProfile.from_json_file(hardware_path)
    graph_dir = Path("configs/graphs")
    graph_paths = sorted(graph_dir.glob("*.json"))
    if not graph_paths:
        raise FileNotFoundError("No sample graphs found in configs/graphs.")
    for graph_path in graph_paths:
        graph = GraphModel.from_json_file(graph_path)
        run_dir = Path(out_dir) / graph.name
        results = run_methods(graph, hardware, methods=methods, max_depth=max_depth)
        write_reports(results, run_dir, with_plots=plots)
        print_summary(results)


def run_graph_workflow(
    graph_path: str | Path,
    hardware_path: str | Path,
    out_dir: str | Path,
    methods: Iterable[str],
    max_depth: int,
    plots: bool,
):
    hardware = HardwareProfile.from_json_file(hardware_path)
    graph = GraphModel.from_json_file(graph_path)
    results = run_methods(graph, hardware, methods=methods, max_depth=max_depth)
    write_reports(results, out_dir, with_plots=plots)
    print_summary(results)
    return results


def run_onnx_workflow(
    onnx_model: str | Path,
    hardware_path: str | Path,
    out_dir: str | Path,
    methods: Iterable[str],
    max_depth: int,
    plots: bool,
    export_method: str,
    fused_onnx_out: str | Path | None = None,
    benchmark_onnxruntime: bool = False,
    benchmark_warmup: int = 10,
    benchmark_repeat: int = 50,
    benchmark_seed: int = 0,
    benchmark_batch_size: int | None = None,
    ort_providers: list[str] | None = None,
    verify_numerical: bool = False,
    verification_backend: str = "reference",
    verification_atol: float = 1e-5,
    verification_rtol: float = 1e-5,
):
    hardware = HardwareProfile.from_json_file(hardware_path)
    imported = load_onnx_model(onnx_model)
    results = run_methods(imported.graph_model, hardware, methods=methods, max_depth=max_depth)
    write_reports(results, out_dir, with_plots=plots)
    plan_by_method = {result.method: result for result in results}
    fused_output = Path(fused_onnx_out) if fused_onnx_out is not None else (Path(out_dir) / f"{export_method}_fused.onnx")
    export_fused_onnx(imported, plan_by_method[export_method], fused_output)
    print_summary(results)
    print(f"Fused ONNX written to: {fused_output}")

    benchmark_result = None
    verification_result = None
    if benchmark_onnxruntime or verify_numerical:
        inputs, input_specs = generate_random_inputs(
            onnx_model,
            seed=benchmark_seed,
            batch_size=benchmark_batch_size,
        )
        providers = list(ort_providers) if ort_providers else ["CPUExecutionProvider"]
        if verify_numerical:
            verification_result = verify_models(
                onnx_model,
                fused_output,
                inputs,
                input_specs,
                backend=verification_backend,
                providers=providers,
                atol=verification_atol,
                rtol=verification_rtol,
            )
            write_verification_report(verification_result, Path(out_dir) / "numerical_verification.json")
            print_verification_summary(verification_result)
        if benchmark_onnxruntime:
            benchmark_result = benchmark_with_onnxruntime(
                onnx_model,
                fused_output,
                inputs,
                input_specs,
                warmup_runs=benchmark_warmup,
                repeat_runs=benchmark_repeat,
                providers=providers,
            )
            write_benchmark_report(benchmark_result, Path(out_dir) / "onnxruntime_benchmark.json")
            print_benchmark_summary(benchmark_result)

    return results, fused_output, benchmark_result, verification_result


def print_summary(results) -> None:
    print(f"\nGraph: {results[0].graph_name} | Hardware: {results[0].hardware_name}")
    print(f"{'method':<10} {'feasible':<10} {'latency_ms':<14} {'kernels':<8}")
    for result in results:
        latency = "inf" if result.estimated_latency_ms == float("inf") else f"{result.estimated_latency_ms:.4f}"
        print(f"{result.method:<10} {str(result.feasible):<10} {latency:<14} {result.kernel_count:<8}")


def print_verification_summary(result) -> None:
    print("\nNumerical verification:")
    print(
        f"backend={result.backend} allclose={result.allclose} "
        f"max_abs_diff={result.max_abs_diff:.8f} mean_abs_diff={result.mean_abs_diff:.8f}"
    )


def print_benchmark_summary(result) -> None:
    print("\nONNX Runtime benchmark:")
    print(
        f"providers={','.join(result.providers)} original_mean_ms={result.original.mean_ms:.4f} "
        f"fused_mean_ms={result.fused.mean_ms:.4f} speedup={result.speedup:.4f}"
    )
