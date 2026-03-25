"""Runtime benchmarking and numerical verification utilities.

This file contains ONNX Runtime timing and original-vs-fused output checking. Add future runtime backends here instead of modifying the core search modules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from ..vendor import import_numpy, import_onnx


@dataclass(frozen=True)
class InputTensorSpec:
    name: str
    shape: list[int]
    dtype: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "runs": self.runs,
        }


@dataclass(frozen=True)
class BenchmarkComparison:
    backend: str
    providers: list[str]
    warmup_runs: int
    repeat_runs: int
    input_specs: list[InputTensorSpec]
    original: BenchmarkStats
    fused: BenchmarkStats
    speedup: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "providers": self.providers,
            "warmup_runs": self.warmup_runs,
            "repeat_runs": self.repeat_runs,
            "input_specs": [item.to_dict() for item in self.input_specs],
            "original": self.original.to_dict(),
            "fused": self.fused.to_dict(),
            "speedup": self.speedup,
        }


@dataclass(frozen=True)
class OutputDiff:
    name: str
    shape: list[int]
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    cosine_similarity: float
    allclose: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "rmse": self.rmse,
            "cosine_similarity": self.cosine_similarity,
            "allclose": self.allclose,
        }


@dataclass(frozen=True)
class VerificationSummary:
    backend: str
    providers: list[str]
    atol: float
    rtol: float
    input_specs: list[InputTensorSpec]
    outputs: list[OutputDiff]
    allclose: bool
    max_abs_diff: float
    mean_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "providers": self.providers,
            "atol": self.atol,
            "rtol": self.rtol,
            "input_specs": [item.to_dict() for item in self.input_specs],
            "outputs": [item.to_dict() for item in self.outputs],
            "allclose": self.allclose,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
        }


def generate_random_inputs(
    model_path: str | Path,
    seed: int = 0,
    batch_size: int | None = None,
) -> tuple[dict[str, Any], list[InputTensorSpec]]:
    np = import_numpy()
    rng = np.random.default_rng(seed)
    specs = get_model_input_specs(model_path, batch_size=batch_size)
    feeds: dict[str, Any] = {}
    for spec in specs:
        dtype = np.dtype(spec.dtype)
        shape = tuple(int(dim) for dim in spec.shape)
        if np.issubdtype(dtype, np.floating):
            value = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            value = rng.integers(0, 17, size=shape, endpoint=False).astype(dtype)
        elif np.issubdtype(dtype, np.bool_):
            value = rng.integers(0, 2, size=shape, endpoint=False).astype(np.bool_)
        else:
            raise NotImplementedError(f"Unsupported benchmark input dtype: {spec.dtype}")
        feeds[spec.name] = value
    return feeds, specs


def get_model_input_specs(model_path: str | Path, batch_size: int | None = None) -> list[InputTensorSpec]:
    model, onnx = _load_onnx_model(model_path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    specs: list[InputTensorSpec] = []
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        shape = _shape_from_value_info(value_info)
        if batch_size is not None and shape:
            shape[0] = int(batch_size)
        elem_type = value_info.type.tensor_type.elem_type
        dtype_name = _dtype_name(elem_type, onnx)
        specs.append(InputTensorSpec(name=value_info.name, shape=shape, dtype=dtype_name))
    return specs


def benchmark_with_onnxruntime(
    original_model: str | Path,
    fused_model: str | Path,
    inputs: dict[str, Any],
    input_specs: Iterable[InputTensorSpec],
    warmup_runs: int = 10,
    repeat_runs: int = 50,
    providers: list[str] | None = None,
) -> BenchmarkComparison:
    original_stats, selected_providers = benchmark_model_with_onnxruntime(
        original_model,
        inputs,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        providers=providers,
    )
    fused_stats, _ = benchmark_model_with_onnxruntime(
        fused_model,
        inputs,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        providers=selected_providers,
    )
    speedup = original_stats.mean_ms / fused_stats.mean_ms if fused_stats.mean_ms > 0 else float("inf")

    return BenchmarkComparison(
        backend="onnxruntime",
        providers=selected_providers,
        warmup_runs=warmup_runs,
        repeat_runs=repeat_runs,
        input_specs=list(input_specs),
        original=original_stats,
        fused=fused_stats,
        speedup=speedup,
    )


def benchmark_model_with_onnxruntime(
    model_path: str | Path,
    inputs: dict[str, Any],
    warmup_runs: int = 10,
    repeat_runs: int = 50,
    providers: list[str] | None = None,
) -> tuple[BenchmarkStats, list[str]]:
    ort = _import_onnxruntime()
    np = import_numpy()
    requested_providers = list(providers) if providers else ["CPUExecutionProvider"]
    selected_providers = _select_providers(ort, requested_providers)

    session = ort.InferenceSession(os.fspath(model_path), providers=selected_providers)
    output_names = [item.name for item in session.get_outputs()]

    for _ in range(max(warmup_runs, 0)):
        session.run(output_names, inputs)

    timings = _measure_session_runs(session, output_names, inputs, repeat_runs)
    return _summarize_timings(timings, np), selected_providers


def verify_models(
    original_model: str | Path,
    fused_model: str | Path,
    inputs: dict[str, Any],
    input_specs: Iterable[InputTensorSpec],
    backend: str = "reference",
    providers: list[str] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> VerificationSummary:
    np = import_numpy()
    if backend == "reference":
        original_outputs = _run_reference_model(original_model, inputs)
        fused_outputs = _run_reference_model(fused_model, inputs)
        used_providers: list[str] = []
    elif backend == "onnxruntime":
        ort = _import_onnxruntime()
        used_providers = _select_providers(ort, list(providers) if providers else ["CPUExecutionProvider"])
        original_outputs = _run_ort_model(original_model, inputs, used_providers)
        fused_outputs = _run_ort_model(fused_model, inputs, used_providers)
    else:
        raise ValueError(f"Unsupported verification backend: {backend}")

    output_names = list(original_outputs.keys())
    if output_names != list(fused_outputs.keys()):
        raise ValueError(
            f"Output mismatch between models: original={list(original_outputs.keys())}, fused={list(fused_outputs.keys())}"
        )

    diffs: list[OutputDiff] = []
    max_abs_diff = 0.0
    total_abs_sum = 0.0
    total_elems = 0
    allclose = True

    for name in output_names:
        baseline = np.asarray(original_outputs[name])
        candidate = np.asarray(fused_outputs[name])
        if baseline.shape != candidate.shape:
            raise ValueError(f"Output '{name}' shape mismatch: {baseline.shape} vs {candidate.shape}")
        abs_diff = np.abs(candidate - baseline)
        output_max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        output_mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(abs_diff)))) if abs_diff.size else 0.0
        cosine_similarity = _cosine_similarity(np, baseline, candidate)
        output_allclose = bool(np.allclose(baseline, candidate, atol=atol, rtol=rtol, equal_nan=True))

        diffs.append(
            OutputDiff(
                name=name,
                shape=[int(dim) for dim in baseline.shape],
                max_abs_diff=output_max_abs,
                mean_abs_diff=output_mean_abs,
                rmse=rmse,
                cosine_similarity=cosine_similarity,
                allclose=output_allclose,
            )
        )
        max_abs_diff = max(max_abs_diff, output_max_abs)
        total_abs_sum += float(abs_diff.sum()) if abs_diff.size else 0.0
        total_elems += int(abs_diff.size)
        allclose = allclose and output_allclose

    mean_abs_diff = total_abs_sum / total_elems if total_elems > 0 else 0.0
    return VerificationSummary(
        backend=backend,
        providers=used_providers,
        atol=atol,
        rtol=rtol,
        input_specs=list(input_specs),
        outputs=diffs,
        allclose=allclose,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )


def run_model_with_onnxruntime(
    model_path: str | Path,
    inputs: dict[str, Any],
    providers: list[str] | None = None,
) -> tuple[list[Any], list[str]]:
    ort = _import_onnxruntime()
    selected_providers = _select_providers(ort, list(providers) if providers else ["CPUExecutionProvider"])
    output_map = _run_ort_model(model_path, inputs, selected_providers)
    return [output_map[name] for name in output_map], selected_providers


def write_benchmark_report(result: BenchmarkComparison, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = [
        "# ONNX Runtime Benchmark",
        "",
        f"- Backend: `{result.backend}`",
        f"- Providers: `{', '.join(result.providers)}`",
        f"- Warmup runs: `{result.warmup_runs}`",
        f"- Repeat runs: `{result.repeat_runs}`",
        "",
        "| Model | Mean (ms) | Median (ms) | P95 (ms) | Std (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| original | {result.original.mean_ms:.4f} | {result.original.median_ms:.4f} | {result.original.p95_ms:.4f} | {result.original.std_ms:.4f} |",
        f"| fused | {result.fused.mean_ms:.4f} | {result.fused.median_ms:.4f} | {result.fused.p95_ms:.4f} | {result.fused.std_ms:.4f} |",
        "",
        f"- Speedup (original / fused): `{result.speedup:.4f}`",
    ]
    output_path.with_suffix('.md').write_text("\n".join(markdown_lines), encoding="utf-8")
    return output_path


def write_verification_report(result: VerificationSummary, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = [
        "# Numerical Verification",
        "",
        f"- Backend: `{result.backend}`",
        f"- Providers: `{', '.join(result.providers) if result.providers else 'N/A'}`",
        f"- atol: `{result.atol}`",
        f"- rtol: `{result.rtol}`",
        f"- Allclose: `{result.allclose}`",
        f"- Global max abs diff: `{result.max_abs_diff:.8f}`",
        f"- Global mean abs diff: `{result.mean_abs_diff:.8f}`",
        "",
        "| Output | Shape | Max Abs Diff | Mean Abs Diff | RMSE | Cosine Similarity | Allclose |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result.outputs:
        markdown_lines.append(
            f"| {item.name} | {item.shape} | {item.max_abs_diff:.8f} | {item.mean_abs_diff:.8f} | {item.rmse:.8f} | {item.cosine_similarity:.8f} | {item.allclose} |"
        )
    output_path.with_suffix('.md').write_text("\n".join(markdown_lines), encoding="utf-8")
    return output_path


def _load_onnx_model(model_path: str | Path) -> tuple[Any, Any]:
    onnx = import_onnx()
    model = onnx.load(Path(model_path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model, onnx


def _shape_from_value_info(value_info: Any) -> list[int]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField('shape'):
        return [1]
    shape: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField('dim_value') and dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        else:
            shape.append(1)
    return shape or [1]


def _dtype_name(elem_type: int, onnx_module: Any) -> str:
    try:
        np_dtype = onnx_module.helper.tensor_dtype_to_np_dtype(elem_type)
        return np_dtype.name
    except Exception:
        fallback = {
            onnx_module.TensorProto.FLOAT16: 'float16',
            onnx_module.TensorProto.FLOAT: 'float32',
            onnx_module.TensorProto.DOUBLE: 'float64',
            onnx_module.TensorProto.INT8: 'int8',
            onnx_module.TensorProto.INT16: 'int16',
            onnx_module.TensorProto.INT32: 'int32',
            onnx_module.TensorProto.INT64: 'int64',
            onnx_module.TensorProto.BOOL: 'bool',
            onnx_module.TensorProto.UINT8: 'uint8',
        }
        return fallback.get(elem_type, 'float32')


def _import_onnxruntime():
    from ..vendor import import_onnxruntime

    return import_onnxruntime()


def _select_providers(ort: Any, requested_providers: list[str]) -> list[str]:
    available = list(ort.get_available_providers())
    selected = [provider for provider in requested_providers if provider in available]
    if selected:
        return selected
    if not available:
        raise RuntimeError('No ONNX Runtime execution providers are available.')
    return [available[0]]


def _measure_session_runs(session: Any, output_names: list[str], inputs: dict[str, Any], repeat_runs: int) -> list[float]:
    timings_ms: list[float] = []
    for _ in range(max(repeat_runs, 1)):
        start = perf_counter()
        session.run(output_names, inputs)
        end = perf_counter()
        timings_ms.append((end - start) * 1e3)
    return timings_ms


def _summarize_timings(timings_ms: list[float], np: Any) -> BenchmarkStats:
    values = np.asarray(timings_ms, dtype=np.float64)
    return BenchmarkStats(
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        std_ms=float(values.std()),
        min_ms=float(values.min()),
        max_ms=float(values.max()),
        runs=int(values.size),
    )


def _run_reference_model(model_path: str | Path, inputs: dict[str, Any]) -> dict[str, Any]:
    model, _ = _load_onnx_model(model_path)
    from onnx.reference import ReferenceEvaluator  # type: ignore

    evaluator = ReferenceEvaluator(model)
    output_names = [value_info.name for value_info in model.graph.output]
    outputs = evaluator.run(None, inputs)
    return {name: value for name, value in zip(output_names, outputs)}


def _run_ort_model(model_path: str | Path, inputs: dict[str, Any], providers: list[str]) -> dict[str, Any]:
    ort = _import_onnxruntime()
    session = ort.InferenceSession(os.fspath(model_path), providers=providers)
    output_names = [item.name for item in session.get_outputs()]
    outputs = session.run(output_names, inputs)
    return {name: value for name, value in zip(output_names, outputs)}


def _cosine_similarity(np: Any, baseline: Any, candidate: Any) -> float:
    baseline_flat = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline_norm = float(np.linalg.norm(baseline_flat))
    candidate_norm = float(np.linalg.norm(candidate_flat))
    if baseline_norm == 0.0 and candidate_norm == 0.0:
        return 1.0
    if baseline_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return float(np.dot(baseline_flat, candidate_flat) / (baseline_norm * candidate_norm))
