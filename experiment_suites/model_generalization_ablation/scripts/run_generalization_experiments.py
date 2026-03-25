from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model generalization and fine-ablation experiment suite.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiment_suites" / "model_generalization_ablation" / "configs" / "model_manifest.json",
        help="Path to the suite manifest.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "model_generalization_ablation",
        help="Directory where experiment outputs will be written.",
    )
    parser.add_argument("--skip-export", action="store_true", help="Do not regenerate torchvision ONNX models before running.")
    parser.add_argument("--force-export", action="store_true", help="Regenerate torchvision ONNX models even if they already exist.")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)

    if not args.skip_export:
        export_module = _load_module(
            ROOT / "experiment_suites" / "model_generalization_ablation" / "scripts" / "export_torchvision_models.py",
            "export_torchvision_models",
        )
        torchvision_models = [
            model_cfg["name"]
            for model_cfg in manifest.get("models", [])
            if model_cfg.get("source") == "torchvision"
        ]
        export_module.export_models(
            model_names=torchvision_models,
            out_dir=(ROOT / "outputs" / "generated_models").resolve(),
            force=bool(args.force_export),
            seed=int(manifest.get("benchmark", {}).get("seed", 0)),
        )

    runner_module = _load_module(
        ROOT / "experiment_suites" / "runtime_ablation_eval" / "scripts" / "run_model_experiments.py",
        "run_model_experiments",
    )
    runner_module.run_experiments(manifest_path, args.out_dir.resolve())


if __name__ == "__main__":
    main()
