# 真实后端融合实验

这个实验目录用于归档以下三类结果：

- 原始 / 融合 ONNX 的 ONNX Runtime 实测结果
- 消融实验结果
- 基于 PyTorch GPU 后端的按融合计划执行实测结果

目录说明：

- `configs/`：本轮实验清单与参数配置
- `scripts/`：实验运行脚本
- `results/`：可提交的关键结果汇总

本轮实验覆盖模型：

- `ResNet50-v2-7`
- `VGG16-12`

关键输出：

- `results/aggregate_ort_results.md`
- `results/aggregate_torch_plan_results.md`
- `results/resnet50/`
- `results/vgg16/`
