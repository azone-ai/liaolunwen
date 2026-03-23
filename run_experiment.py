from __future__ import annotations

from pathlib import Path

from fusion_lab.cli import main


def main_local() -> None:
    root = Path(__file__).resolve().parent

    # 选择要测试的 ONNX 模型。
    # 这里默认使用 ResNet50；如果你想换成 VGG16，把下面这一行改掉即可。
    onnx_model = root / "model" / "resnet50-v2-7.onnx"
    # onnx_model = root / "model" / "vgg16-12.onnx"

    # 硬件配置文件路径。
    hardware = root / "configs" / "hardware" / "generic_gpu.json"

    # 输出目录。
    # 程序会在这里写入 summary、plan.json 以及融合后的 onnx 文件。
    out_dir = root / "outputs" / "resnet50_local_run"

    # 选择导出融合后 ONNX 时使用的方法。
    export_method = "hw_aware"

    # 想跑哪些方法，就在这里改列表。
    methods = ["none", "hw_aware"]

    # 动态规划允许跨越的最大层数。
    max_depth = 4

    # 是否额外生成延迟对比图。
    enable_plots = False

    # 如果你想自己指定融合后 onnx 的文件名，就改这里。
    # 不想手动指定的话，也可以设成 None，让程序自动生成。
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
    main_local()
