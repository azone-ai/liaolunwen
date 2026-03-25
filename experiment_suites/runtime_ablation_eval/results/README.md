# 结果归档说明

这个目录保存的是可以直接提交到仓库的结果摘要，而不是 `outputs/` 目录中的全部临时实验产物。

包含内容：

- `aggregate_ort_results.*`：ONNX Runtime 真实运行结果聚合
- `aggregate_torch_plan_results.*`：按融合计划在 PyTorch GPU 后端执行的真实结果聚合
- `resnet50/`：ResNet50 的分模型结果
- `vgg16/`：VGG16 的分模型结果

分模型目录中包含：

- `combined_results_ort.*`：ONNX Runtime 结果
- `hardware_analysis.*`：occupancy / registers / shared memory 汇总
- `torch_plan_backend.*`：按融合计划执行的真实 GPU 结果
