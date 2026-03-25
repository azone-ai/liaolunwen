# 更多模型与细消融实验

这个实验套件用于补齐论文后续最需要的两块内容：

- 更多模型泛化：在现有 `ResNet50`、`VGG16` 之外，再加入 `AlexNet`、`MobileNetV2`
- 更细粒度消融：把硬件感知惩罚拆成可单独关闭的版本，分别分析线程搜索、几何匹配、寄存器、共享内存和指令缓存项

## 当前实验清单

### 可直接运行的模型

| 模型 | 来源 | 目的 | 当前状态 |
| --- | --- | --- | --- |
| ResNet50 | 仓库已有 ONNX | 真实大模型基线 | 已可运行 |
| VGG16 | 仓库已有 ONNX | 真实大模型基线 | 已可运行 |
| AlexNet | `torchvision` 导出 | 补充经典 CNN | 已接入导出脚本 |
| MobileNetV2 | `torchvision` 导出 | 补充轻量网络 / 深度可分离卷积 | 已接入导出脚本 |

### 下一阶段建议模型

| 模型 | 价值 | 当前阻塞 |
| --- | --- | --- |
| DenseNet121 | 能体现 `Concat` 型结构的融合限制 | Torch 后端还未支持 `Concat` |
| Transformer Encoder / BERT Block | 能补齐注意力 / FFN 结构 | 需要稳定的可导出 ONNX 模型清单 |
| UNet Block | 能补齐 skip connection + concat 场景 | 需要 `Concat` 和更复杂的图模式支持 |

### 细消融方法

| 方法名 | 含义 |
| --- | --- |
| `dp_paper` | 去掉全部软惩罚，只保留 DP 搜索 |
| `hw_no_thread_search` | 保留硬件惩罚，但不搜索候选线程块规模 |
| `hw_no_geometry` | 去掉线程几何不匹配惩罚 |
| `hw_no_register` | 去掉寄存器压力惩罚 |
| `hw_no_shared_memory` | 去掉共享内存压力惩罚 |
| `hw_no_icache` | 去掉指令缓存压力惩罚 |
| `hw_aware` | 完整硬件感知方法 |

## 推荐运行顺序

1. 先导出 `torchvision` 模型
2. 跑解析式结果 + ONNX Runtime 正确性
3. 再跑 Torch 计划执行后端的真实结果
4. 最后把 `AlexNet`、`MobileNetV2` 纳入总表

## 常用命令

导出新增模型：

```powershell
python experiment_suites/model_generalization_ablation/scripts/export_torchvision_models.py
```

运行整个实验套件：

```powershell
python experiment_suites/model_generalization_ablation/scripts/run_generalization_experiments.py
```

只跑已经导出的模型，不重复导出：

```powershell
python experiment_suites/model_generalization_ablation/scripts/run_generalization_experiments.py --skip-export
```

## 输出位置

- 新增导出的模型：`outputs/generated_models/`
- 实验结果：`outputs/model_generalization_ablation/`
- 配置入口：[configs/model_manifest.json](D:/小论文/codex_xyx/experiment_suites/model_generalization_ablation/configs/model_manifest.json)

## 维护建议

- 新增模型时，优先只改 `configs/model_manifest.json`
- 如果是 `torchvision` 模型，优先补进导出脚本注册表
- 如果新模型报算子不支持，优先改 `fusion_lab/research/torch_plan_backend.py`
- 如果只是新增一轮论文实验，不要改这个套件，直接复制一份到新的实验目录
