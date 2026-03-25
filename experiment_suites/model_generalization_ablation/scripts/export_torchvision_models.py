from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchvision.models as tv_models

from fusion_lab.vendor import import_onnx


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "alexnet": {
        "factory": tv_models.alexnet,
        "input_shape": (1, 3, 224, 224),
        "opset_version": 18,
    },
    "mobilenet_v2": {
        "factory": tv_models.mobilenet_v2,
        "input_shape": (1, 3, 224, 224),
        "opset_version": 18,
    },
}


def _summarize_onnx_ops(model_path: Path) -> dict[str, int]:
    onnx = import_onnx()
    model = onnx.load(model_path)
    counter = Counter(node.op_type for node in model.graph.node)
    return dict(sorted(counter.items()))


def export_models(
    model_names: list[str],
    out_dir: Path,
    force: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported torchvision export target: {model_name}")

        spec = MODEL_REGISTRY[model_name]
        output_path = out_dir / f"{model_name}.onnx"
        if output_path.exists() and not force:
            ops = _summarize_onnx_ops(output_path)
            rows.append(
                {
                    "model": model_name,
                    "path": str(output_path.relative_to(ROOT)),
                    "status": "reused",
                    "ops": ops,
                }
            )
            continue

        model = spec["factory"](weights=None)
        model.eval()
        sample_input = torch.randn(*spec["input_shape"])

        try:
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=int(spec.get("opset_version", 18)),
                dynamo=True,
                external_data=False,
            )
            export_mode = "dynamo"
        except Exception:
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=int(spec.get("opset_version", 18)),
                dynamo=False,
            )
            export_mode = "legacy"

        ops = _summarize_onnx_ops(output_path)
        rows.append(
            {
                "model": model_name,
                "path": str(output_path.relative_to(ROOT)),
                "status": f"exported:{export_mode}",
                "ops": ops,
            }
        )

    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = [
        "# torchvision ONNX 导出结果",
        "",
        "| 模型 | 状态 | 路径 | 主要算子 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        ops_text = ", ".join(f"{key}:{value}" for key, value in row["ops"].items())
        markdown_lines.append(f"| {row['model']} | {row['status']} | {row['path']} | {ops_text} |")
    summary_path.with_suffix(".md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export additional torchvision models to ONNX for generalization experiments.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
        help="torchvision model names to export.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "generated_models",
        help="Directory where exported ONNX models will be stored.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing exported models.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for dummy export inputs.")
    args = parser.parse_args()

    export_models(
        model_names=list(args.models),
        out_dir=args.out_dir.resolve(),
        force=args.force,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
