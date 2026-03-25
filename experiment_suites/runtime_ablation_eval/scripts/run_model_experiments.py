from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion_lab.hardware import HardwareProfile
from fusion_lab.onnx_bridge import export_fused_onnx, load_onnx_model
from fusion_lab.research.experiment_runner import run_methods, write_reports
from fusion_lab.research.runtime_validation import (
    benchmark_model_with_onnxruntime,
    generate_random_inputs,
    verify_models,
    write_verification_report,
)
from fusion_lab.research.torch_plan_backend import (
    benchmark_torch_plan,
    run_torch_plan_outputs,
    verify_torch_plan_against_onnx,
    write_torch_backend_report,
)
from fusion_lab.vendor import import_onnxruntime


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_row(result: Any) -> dict[str, Any]:
    return {
        "method": result.method,
        "estimated_latency_ms": result.estimated_latency_ms,
        "kernel_count": result.kernel_count,
        "search_time_ms": result.search_time_ms,
        "avg_occupancy": result.avg_occupancy,
        "avg_registers_per_thread": result.avg_registers_per_thread,
        "max_registers_per_thread": result.max_registers_per_thread,
        "avg_shared_mem_kib": result.avg_shared_mem_bytes / 1024.0,
        "max_shared_mem_kib": result.max_shared_mem_bytes / 1024.0,
        "avg_threads_per_block": result.avg_threads_per_block,
        "max_threads_per_block": result.max_threads_per_block,
        "avg_penalty_multiplier": result.avg_penalty_multiplier,
        "feasible": result.feasible,
    }


def _write_markdown(rows: list[dict[str, Any]], title: str, output_path: Path) -> None:
    lines = [
        f"# {title}",
        "",
        "| Model | Method | Feasible | Est. Latency (ms) | Runtime Mean (ms) | Runtime Speedup | Search Time (ms) | Kernels | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Avg Threads/Block | Allclose | Max Abs Diff |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        runtime_mean = "-" if row["runtime_mean_ms"] is None else f"{row['runtime_mean_ms']:.4f}"
        runtime_speedup = "-" if row["runtime_speedup"] is None else f"{row['runtime_speedup']:.4f}"
        max_abs_diff = "-" if row["max_abs_diff"] is None else f"{row['max_abs_diff']:.8f}"
        allclose = "-" if row["allclose"] is None else str(row["allclose"])
        lines.append(
            f"| {row['model_name']} | {row['method']} | {row['feasible']} | {row['estimated_latency_ms']:.4f} | {runtime_mean} | {runtime_speedup} | {row['search_time_ms']:.3f} | {row['kernel_count']} | {row['avg_occupancy']:.3f} | {row['avg_registers_per_thread']:.2f} | {row['avg_shared_mem_kib']:.3f} | {row['avg_threads_per_block']:.2f} | {allclose} | {max_abs_diff} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _write_json_and_csv(rows: list[dict[str, Any]], out_dir: Path, stem: str) -> None:
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    with open(out_dir / f"{stem}.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _run_ort_outputs(model_path: Path, feeds: dict[str, Any], providers: list[str]) -> list[Any]:
    ort = import_onnxruntime()
    session = ort.InferenceSession(str(model_path), providers=providers)
    output_names = [item.name for item in session.get_outputs()]
    return session.run(output_names, feeds)


def run_experiments(manifest_path: Path, out_dir: Path) -> None:
    manifest = _load_manifest(manifest_path)
    hardware = HardwareProfile.from_json_file(ROOT / manifest["hardware_path"])
    methods = manifest["methods"]
    max_depth = int(manifest.get("max_depth", 4))
    benchmark_cfg = manifest["benchmark"]
    torch_cfg = manifest.get("torch_backend", {})
    runtime_methods = list(manifest.get("runtime_methods", methods))
    verification_cfg = manifest["verification"]
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_torch_rows: list[dict[str, Any]] = []

    for model_cfg in manifest["models"]:
        model_name = model_cfg["name"]
        model_path = ROOT / model_cfg["path"]
        model_out_dir = out_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        imported = load_onnx_model(model_path)
        results = run_methods(imported.graph_model, hardware, methods=methods, max_depth=max_depth)
        write_reports(results, model_out_dir, with_plots=False)
        result_by_method = {result.method: result for result in results}

        inputs, input_specs = generate_random_inputs(
            model_path,
            seed=int(benchmark_cfg.get("seed", 0)),
        )
        original_stats, selected_providers = benchmark_model_with_onnxruntime(
            model_path,
            inputs,
            warmup_runs=int(benchmark_cfg.get("warmup_runs", 10)),
            repeat_runs=int(benchmark_cfg.get("repeat_runs", 30)),
            providers=list(benchmark_cfg.get("providers", ["CPUExecutionProvider"])),
        )

        model_rows: list[dict[str, Any]] = []
        none_row = _summary_row(result_by_method["none"])
        none_row.update(
            {
                "model_name": model_name,
                "runtime_mean_ms": original_stats.mean_ms,
                "runtime_speedup": 1.0,
                "runtime_median_ms": original_stats.median_ms,
                "runtime_p95_ms": original_stats.p95_ms,
                "runtime_std_ms": original_stats.std_ms,
                "allclose": True,
                "max_abs_diff": 0.0,
            }
        )
        model_rows.append(none_row)

        for method in methods:
            if method == "none":
                continue
            fused_path = model_out_dir / f"{method}_fused.onnx"
            export_fused_onnx(imported, result_by_method[method], fused_path)
            verification = verify_models(
                model_path,
                fused_path,
                inputs,
                input_specs,
                backend=verification_cfg.get("backend", "onnxruntime"),
                providers=selected_providers,
                atol=float(verification_cfg.get("atol", 1e-5)),
                rtol=float(verification_cfg.get("rtol", 1e-5)),
            )
            write_verification_report(verification, model_out_dir / f"{method}_numerical_verification.json")

            fused_stats, _ = benchmark_model_with_onnxruntime(
                fused_path,
                inputs,
                warmup_runs=int(benchmark_cfg.get("warmup_runs", 10)),
                repeat_runs=int(benchmark_cfg.get("repeat_runs", 30)),
                providers=selected_providers,
            )
            row = _summary_row(result_by_method[method])
            row.update(
                {
                    "model_name": model_name,
                    "runtime_mean_ms": fused_stats.mean_ms,
                    "runtime_speedup": original_stats.mean_ms / fused_stats.mean_ms if fused_stats.mean_ms > 0 else float("inf"),
                    "runtime_median_ms": fused_stats.median_ms,
                    "runtime_p95_ms": fused_stats.p95_ms,
                    "runtime_std_ms": fused_stats.std_ms,
                    "allclose": verification.allclose,
                    "max_abs_diff": verification.max_abs_diff,
                }
            )
            model_rows.append(row)

        _write_json_and_csv(model_rows, model_out_dir, "combined_results")
        _write_markdown(model_rows, f"{model_name} Combined Experiment Results", model_out_dir / "combined_results.md")
        aggregate_rows.extend(model_rows)

        if bool(torch_cfg.get("enabled", False)):
            ort_reference_outputs = _run_ort_outputs(model_path, inputs, selected_providers)
            torch_rows: list[dict[str, Any]] = []
            for method in runtime_methods:
                plan = result_by_method[method]
                stats, compiled_module = benchmark_torch_plan(
                    imported,
                    plan,
                    inputs,
                    device=torch_cfg.get("device", "cuda"),
                    warmup_runs=int(torch_cfg.get("warmup_runs", 10)),
                    repeat_runs=int(torch_cfg.get("repeat_runs", 30)),
                    compile_model=bool(torch_cfg.get("compile_model", True)),
                    backend=str(torch_cfg.get("backend", "auto")),
                )
                outputs = run_torch_plan_outputs(
                    compiled_module,
                    inputs,
                    device=torch_cfg.get("device", "cuda"),
                )
                verification = verify_torch_plan_against_onnx(
                    ort_reference_outputs,
                    outputs,
                    atol=float(torch_cfg.get("atol", 1e-4)),
                    rtol=float(torch_cfg.get("rtol", 1e-4)),
                )
                row = _summary_row(plan)
                row.update(
                    {
                        "model_name": model_name,
                        "runtime_backend": "torch_compile",
                        "torch_backend_mode": compiled_module.mode,
                        "runtime_mean_ms": stats.mean_ms,
                        "runtime_median_ms": stats.median_ms,
                        "runtime_p95_ms": stats.p95_ms,
                        "runtime_std_ms": stats.std_ms,
                        "runtime_speedup": None,
                        "compile_time_ms": stats.compile_time_ms,
                        "allclose": verification.allclose,
                        "max_abs_diff": verification.max_abs_diff,
                        "mean_abs_diff": verification.mean_abs_diff,
                        "cosine_similarity": verification.cosine_similarity,
                    }
                )
                torch_rows.append(row)

            baseline_runtime = next(row["runtime_mean_ms"] for row in torch_rows if row["method"] == "none")
            for row in torch_rows:
                row["runtime_speedup"] = baseline_runtime / row["runtime_mean_ms"] if row["runtime_mean_ms"] > 0 else float("inf")

            _write_json_and_csv(torch_rows, model_out_dir, "torch_plan_backend")
            write_torch_backend_report(torch_rows, model_out_dir / "torch_plan_backend.json")
            aggregate_torch_rows.extend(torch_rows)

    _write_json_and_csv(aggregate_rows, out_dir, "aggregate_results")
    _write_markdown(aggregate_rows, "Aggregate Combined Experiment Results", out_dir / "aggregate_results.md")
    if aggregate_torch_rows:
        _write_json_and_csv(aggregate_torch_rows, out_dir, "aggregate_torch_plan_backend")
        write_torch_backend_report(aggregate_torch_rows, out_dir / "aggregate_torch_plan_backend.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run runtime benchmark, ablation and hardware analysis experiments.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiment_suites" / "runtime_ablation_eval" / "configs" / "model_manifest.json",
        help="Path to the experiment manifest JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "runtime_ablation_eval",
        help="Directory where experiment outputs will be written.",
    )
    args = parser.parse_args()

    run_experiments(args.manifest.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
