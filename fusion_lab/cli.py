from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import DEFAULT_METHODS, run_methods, write_reports
from .graph import GraphModel
from .hardware import HardwareProfile
from .onnx_bridge import export_fused_onnx, load_onnx_model


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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.graph is not None and args.onnx_model is not None:
        parser.error("--graph and --onnx-model are mutually exclusive.")

    if args.all_samples:
        if args.onnx_model is not None:
            parser.error("--all-samples only supports JSON sample graphs, not --onnx-model.")
        if args.hardware is None:
            parser.error("--hardware is required when --all-samples is used.")
        hardware = HardwareProfile.from_json_file(args.hardware)
        graph_dir = Path("configs/graphs")
        graph_paths = sorted(graph_dir.glob("*.json"))
        if not graph_paths:
            parser.error("No sample graphs found in configs/graphs.")
        for graph_path in graph_paths:
            graph = GraphModel.from_json_file(graph_path)
            run_dir = args.out_dir / graph.name
            results = run_methods(graph, hardware, methods=args.methods, max_depth=args.max_depth)
            write_reports(results, run_dir, with_plots=args.plots)
            _print_summary(results)
        return

    if args.hardware is None:
        parser.error("--hardware is required unless --all-samples is used.")

    hardware = HardwareProfile.from_json_file(args.hardware)
    if args.onnx_model is not None:
        if args.export_method not in args.methods:
            parser.error("--export-method must also be included in --methods.")
        imported = load_onnx_model(args.onnx_model)
        results = run_methods(imported.graph_model, hardware, methods=args.methods, max_depth=args.max_depth)
        write_reports(results, args.out_dir, with_plots=args.plots)
        plan_by_method = {result.method: result for result in results}
        fused_output = args.fused_onnx_out or (args.out_dir / f"{args.export_method}_fused.onnx")
        export_fused_onnx(imported, plan_by_method[args.export_method], fused_output)
        _print_summary(results)
        print(f"Fused ONNX written to: {fused_output}")
        return

    if args.graph is None:
        parser.error("Either --graph or --onnx-model is required.")

    graph = GraphModel.from_json_file(args.graph)
    results = run_methods(graph, hardware, methods=args.methods, max_depth=args.max_depth)
    write_reports(results, args.out_dir, with_plots=args.plots)
    _print_summary(results)


def _print_summary(results) -> None:
    print(f"\nGraph: {results[0].graph_name} | Hardware: {results[0].hardware_name}")
    print(f"{'method':<10} {'feasible':<10} {'latency_ms':<14} {'kernels':<8}")
    for result in results:
        latency = "inf" if result.estimated_latency_ms == float("inf") else f"{result.estimated_latency_ms:.4f}"
        print(f"{result.method:<10} {str(result.feasible):<10} {latency:<14} {result.kernel_count:<8}")


if __name__ == "__main__":
    main()
