# 实验备注

## 本轮目标

- 扩模型：`AlexNet`、`MobileNetV2`
- 扩消融：`hw_no_geometry`、`hw_no_register`、`hw_no_shared_memory`、`hw_no_icache`
- 保持现有主流程不大改，优先复用已有 runner

## 当前已知限制

- `DenseNet121` 等含 `Concat` 的模型暂未纳入真实后端实验
- Windows + CUDA 环境下 `torch.compile(inductor)` 可能退回到 `trace` 模式
- 新增模型默认导出到 `outputs/generated_models/`，不直接提交大模型文件到仓库

## 后续建议

1. 跑完 `AlexNet`、`MobileNetV2` 后，先做一版总表
2. 再决定是否补 `DenseNet121`
3. 如果要补 `DenseNet121`，优先加 `Concat` 支持而不是继续堆新模型
