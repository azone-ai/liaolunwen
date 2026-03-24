"""Convenience launcher for local experiments.

Use this file for quick local runs. For automation or scripts, prefer `python -m fusion_lab.cli ...`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fusion_lab.cli import main


def main_local() -> None:
    root = Path(__file__).resolve().parent

    # Default local smoke-test model: ResNet50.
    onnx_model = root / "model" / "resnet50-v2-7.onnx"
    # onnx_model = root / "model" / "vgg16-12.onnx"

    hardware = root / "configs" / "hardware" / "generic_gpu.json"
    out_dir = root / "outputs" / "resnet50_local_run"
    export_method = "hw_aware"
    methods = ["none", "hw_aware"]
    max_depth = 4
    enable_plots = False
    fused_onnx_out = out_dir / "resnet50_hw_aware_fused.onnx"
    # fused_onnx_out = None

    argv = [
        "--onnx-model", str(onnx_model),
        "--hardware", str(hardware),
        "--out-dir", str(out_dir),
        "--methods", *methods,
        "--export-method", export_method,
        "--max-depth", str(max_depth),
    ]

    if enable_plots:
        argv.append("--plots")

    if fused_onnx_out is not None:
        argv.extend(["--fused-onnx-out", str(fused_onnx_out)])

    main(argv)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main_local()
