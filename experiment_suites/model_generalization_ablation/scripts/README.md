# 脚本说明

- `export_torchvision_models.py`
  - 导出 `AlexNet`、`MobileNetV2` 等 `torchvision` 模型到 `outputs/generated_models/`
  - 适合在第一次跑实验前执行

- `run_generalization_experiments.py`
  - 本实验套件总入口
  - 默认会先检查并导出缺失的 `torchvision` 模型，然后复用现有实验 runner 执行完整实验

如果只是想快速补一个新模型，建议先改导出脚本注册表，再改 manifest。
