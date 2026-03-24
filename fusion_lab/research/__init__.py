"""Experiment-only utilities: report generation, runtime validation and future study code."""

from .experiment_runner import DEFAULT_METHODS, run_methods, write_reports
from .runtime_validation import (
    BenchmarkComparison,
    BenchmarkStats,
    InputTensorSpec,
    OutputDiff,
    VerificationSummary,
    benchmark_with_onnxruntime,
    generate_random_inputs,
    get_model_input_specs,
    verify_models,
    write_benchmark_report,
    write_verification_report,
)

__all__ = [
    "DEFAULT_METHODS",
    "run_methods",
    "write_reports",
    "BenchmarkComparison",
    "BenchmarkStats",
    "InputTensorSpec",
    "OutputDiff",
    "VerificationSummary",
    "benchmark_with_onnxruntime",
    "generate_random_inputs",
    "get_model_input_specs",
    "verify_models",
    "write_benchmark_report",
    "write_verification_report",
]
