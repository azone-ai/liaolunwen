"""Thin command-line interface.

This module only parses arguments and dispatches work to workflow helpers.
Keep business logic out of this file so future experiments can reuse the same pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .research.experiment_runner import DEFAULT_METHODS
from .workflow.pipeline import run_all_sample_workflow, run_graph_workflow, run_onnx_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hardware-aware fusion experiments.")
    parser.add_argument("--graph", type=Path, help="Path to the graph JSON file.")
    parser.add_argument("--onnx-model", type=Path, help="Path to the input ONNX model.")
    parser.add_argument("--hardware", type=Path, help="Path to the hardware JSON file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for reports.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(DEFAULT_METHODS),
        help="Methods to run.",
    )
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum consecutive layer span for DP.")
    parser.add_argument("--plots", action="store_true", help="Emit a latency comparison plot.")
    parser.add_argument(
        "--export-method",
        default="hw_aware",
        choices=list(DEFAULT_METHODS),
        help="Method whose plan should be exported as fused ONNX.",
    )
    parser.add_argument(
        "--fused-onnx-out",
        type=Path,
        help="Path for the fused ONNX model. Defaults to <out-dir>/<method>_fused.onnx.",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Run every graph in configs/graphs against the provided hardware profile.",
    )
    parser.add_argument(
        "--benchmark-onnxruntime",
        action="store_true",
        help="Benchmark original and fused ONNX models with ONNX Runtime.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=10, help="Number of warmup runs before ONNX Runtime timing.")
    parser.add_argument("--benchmark-repeat", type=int, default=50, help="Number of timed runs for ONNX Runtime benchmarking.")
    parser.add_argument("--benchmark-seed", type=int, default=0, help="Random seed used to generate ONNX benchmark inputs.")
    parser.add_argument("--benchmark-batch-size", type=int, help="Override the first input dimension when generating benchmark inputs.")
    parser.add_argument(
        "--ort-providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Preferred ONNX Runtime execution providers, in priority order.",
    )
    parser.add_argument("--verify-numerical", action="store_true", help="Verify numerical closeness between original and fused ONNX models.")
    parser.add_argument(
        "--verification-backend",
        choices=["reference", "onnxruntime"],
        default="reference",
        help="Execution backend used for numerical verification.",
    )
    parser.add_argument("--verification-atol", type=float, default=1e-5, help="Absolute tolerance for output checks.")
    parser.add_argument("--verification-rtol", type=float, default=1e-5, help="Relative tolerance for output checks.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.graph is not None and args.onnx_model is not None:
        parser.error("--graph and --onnx-model are mutually exclusive.")

    if args.all_samples:
        if args.onnx_model is not None:
            parser.error("--all-samples only supports JSON sample graphs, not --onnx-model.")
        if args.benchmark_onnxruntime or args.verify_numerical:
            parser.error("--benchmark-onnxruntime and --verify-numerical require --onnx-model.")
        if args.hardware is None:
            parser.error("--hardware is required when --all-samples is used.")
        run_all_sample_workflow(
            hardware_path=args.hardware,
            out_dir=args.out_dir,
            methods=args.methods,
            max_depth=args.max_depth,
            plots=args.plots,
        )
        return

    if args.hardware is None:
        parser.error("--hardware is required unless --all-samples is used.")

    if args.onnx_model is not None:
        if args.export_method not in args.methods:
            parser.error("--export-method must also be included in --methods.")
        run_onnx_workflow(
            onnx_model=args.onnx_model,
            hardware_path=args.hardware,
            out_dir=args.out_dir,
            methods=args.methods,
            max_depth=args.max_depth,
            plots=args.plots,
            export_method=args.export_method,
            fused_onnx_out=args.fused_onnx_out,
            benchmark_onnxruntime=args.benchmark_onnxruntime,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_repeat=args.benchmark_repeat,
            benchmark_seed=args.benchmark_seed,
            benchmark_batch_size=args.benchmark_batch_size,
            ort_providers=args.ort_providers,
            verify_numerical=args.verify_numerical,
            verification_backend=args.verification_backend,
            verification_atol=args.verification_atol,
            verification_rtol=args.verification_rtol,
        )
        return

    if args.benchmark_onnxruntime or args.verify_numerical:
        parser.error("--benchmark-onnxruntime and --verify-numerical require --onnx-model.")

    if args.graph is None:
        parser.error("Either --graph or --onnx-model is required.")

    run_graph_workflow(
        graph_path=args.graph,
        hardware_path=args.hardware,
        out_dir=args.out_dir,
        methods=args.methods,
        max_depth=args.max_depth,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
