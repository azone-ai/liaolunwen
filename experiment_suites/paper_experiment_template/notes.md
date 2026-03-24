# 实验记录模板

## 实验目标

说明这一轮实验想回答什么问题。

示例：

- 硬件感知惩罚是否改善了 edge GPU 场景下的融合选择。
- ONNX Runtime 的真实加速比与解析式估计的接近程度如何。

## 实验矩阵

把计划运行的组合列在这里。

| 模型 | 硬件 | 方法 | Batch | 备注 |
| --- | --- | --- | --- | --- |
| example | example | none, dp_paper, hw_aware | 1 | 替换这一行 |

## 使用过的命令

把实际执行过的命令记录在这里，方便复现。

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/example ^
  --methods none hw_aware ^
  --export-method hw_aware
```

## 关键结论

实验结束后，把结论简短写在这里。

- 结论 1：
- 结论 2：
- 结论 3：

## 后续事项

- 还有哪些实验需要补跑。
- 这一轮实验要支撑哪一张图或哪一张表。
