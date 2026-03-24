# 面向硬件感知的算子融合实验框架

## 项目简介

本仓库是一个用于论文研究与方法验证的算子融合实验框架，核心方法链包括：

- 算子映射类型分类
- 融合合法性规则判断
- 硬件感知代价建模
- 基于动态规划的全局融合搜索

当前项目定位为研究型实验平台，而不是 TVM 生产级 pass。它主要服务于：

- 方法设计验证
- 消融实验
- 实验报告输出
- ONNX 导入与融合后导出
- 数值一致性验证
- ONNX Runtime 真实延迟基准测试

## 当前能力

### 核心融合实验能力

- 支持 `none`、`greedy`、`dp_paper`、`hw_aware` 四类融合方法
- 支持基于硬件感知 roofline 风格模型的解析式延迟估计
- 支持导出每种方法的融合计划 JSON
- 支持输出 JSON、CSV、Markdown 格式的实验汇总
- 支持生成延迟对比图

### ONNX 工作流能力

- 支持将 ONNX 模型导入内部图表示
- 支持将融合计划重新导出为 ONNX 模型
- 支持原始模型与融合模型的数值一致性验证
- 支持使用 ONNX Runtime 测试原始模型与融合模型的真实运行延迟

### 面向论文实验的支持能力

- 已将可复用核心逻辑与实验专用代码分层
- 已预留 `experiment_suites/` 目录用于后续论文实验
- 已包含核心流程、ONNX 桥接与运行时验证的回归测试

## 项目结构

### 一、可复用核心层

- `fusion_lab/graph.py`：内部图结构、拓扑辅助函数、区间切分辅助逻辑
- `fusion_lab/ops.py`：算子类型识别、算子级估计辅助函数
- `fusion_lab/fusion_rules.py`：融合合法性规则
- `fusion_lab/cost_model.py`：硬件感知延迟与资源代价模型
- `fusion_lab/search.py`：`none`、`greedy`、`dp` 搜索与结果组织
- `fusion_lab/hardware.py`：硬件配置抽象
- `fusion_lab/enums.py`：共享枚举定义
- `fusion_lab/vendor.py`：本地可选依赖加载工具
- `fusion_lab/core/`：核心能力的统一导出入口

### 二、流程编排层

- `fusion_lab/cli.py`：命令行参数解析与流程分发
- `fusion_lab/workflow/pipeline.py`：端到端流程编排
- `fusion_lab/onnx_bridge.py`：ONNX 导入导出桥接
- `run_experiment.py`：本地便捷启动脚本

### 三、实验与验证层

- `fusion_lab/research/experiment_runner.py`：实验执行与结果汇总输出
- `fusion_lab/research/runtime_validation.py`：数值验证与 ONNX Runtime 基准测试
- `fusion_lab/experiments.py`：向后兼容包装
- `fusion_lab/runtime_benchmark.py`：向后兼容包装
- `tests/`：回归测试

### 四、输入输出与实验目录

- `configs/graphs/`：样例图配置
- `configs/hardware/`：硬件配置文件
- `model/`：ONNX 模型
- `outputs/`：实验输出、导出模型与报告
- `.vendor/`：本地可选依赖，例如 `onnxruntime`
- `experiment_suites/`：未来每一轮论文实验的独立目录

如果你希望按文件进一步了解代码职责，建议继续阅读：

- `PROJECT_LAYOUT.md`
- `CODEBASE_GUIDE.md`

## 方法概览

### 算子映射类型

框架使用映射类型来描述算子的执行特征，目前包括：

- `ONE_TO_ONE`
- `ONE_TO_MANY`
- `MANY_TO_MANY`
- `REORGANIZE`
- `SHUFFLE`
- `REDUCTION`
- `OPAQUE`

这里显式区分了 `REDUCTION`，因为像 `ReduceSum`、`LayerNorm` 这类算子在融合合法性和硬件代价建模上，与普通逐点算子或常规计算算子差异明显。

### 硬件感知代价模型

框架先使用 roofline 风格的基础代价估计融合块延迟：

```text
T_base(B, h) = max(F(B) / (eta_c(B, h) * PeakFLOPS(h)),
                   M(B) / (eta_m(B, h) * Bandwidth(h)))
               + T_launch(h)
```

最终执行时间估计为：

```text
T_exec(B, h) = T_base(B, h) * Penalty(B, h)
```

其中 `Penalty(B, h)` 用于描述融合后仍然合法、但可能带来风险的硬件压力，例如：

- 寄存器占用过高
- 共享内存占用过高
- 线程块几何不匹配
- 指令缓存压力偏大

此外，框架同时使用：

- 硬约束：用于直接剪枝不可执行的融合候选
- 软惩罚：用于降低高风险但仍可执行候选的优先级

### 全局搜索策略

框架不是只做局部相邻算子融合，而是基于层区间做动态规划搜索：

```text
dp[i] = min_j (dp[j - 1] + C(L_j ... L_i))
```

这使得框架能够统一比较：

- 不融合
- 局部贪心融合
- 不含硬件惩罚的全局 DP 融合
- 含硬件感知惩罚的全局 DP 融合

## 运行方式

### 1. 运行单个样例图

```bash
python run_experiment.py ^
  --graph configs/graphs/residual_block.json ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/residual_block
```

### 2. 运行全部样例图

```bash
python -m fusion_lab.cli ^
  --all-samples ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/all_samples
```

### 3. 运行 ONNX 模型并导出融合后的 ONNX

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_run ^
  --methods none hw_aware ^
  --export-method hw_aware
```

### 4. 验证原始 ONNX 与融合 ONNX 的数值一致性

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_verify ^
  --methods none hw_aware ^
  --export-method hw_aware ^
  --verify-numerical ^
  --verification-backend onnxruntime
```

如果你只想使用依赖更少的参考后端，也可以改为：

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_verify_reference ^
  --methods none hw_aware ^
  --export-method hw_aware ^
  --verify-numerical ^
  --verification-backend reference
```

### 5. 测试 ONNX Runtime 真实延迟

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_bench ^
  --methods none hw_aware ^
  --export-method hw_aware ^
  --benchmark-onnxruntime ^
  --benchmark-warmup 10 ^
  --benchmark-repeat 50
```

### 6. 同时做数值验证和真实延迟基准测试

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_full_eval ^
  --methods none hw_aware ^
  --export-method hw_aware ^
  --verify-numerical ^
  --verification-backend onnxruntime ^
  --benchmark-onnxruntime ^
  --benchmark-warmup 10 ^
  --benchmark-repeat 50
```

### 7. 运行测试

```bash
python -m pytest tests
```

## 输出文件说明

### 搜索与汇总输出

每次实验通常会生成：

- `summary.json`
- `summary.csv`
- `summary.md`
- `<method>_plan.json`
- `latency_comparison.png`，当启用 `--plots` 时生成

### ONNX 工作流输出

运行 ONNX 相关流程时，可能额外生成：

- `<method>_fused.onnx`
- `numerical_verification.json`
- `numerical_verification.md`
- `onnxruntime_benchmark.json`
- `onnxruntime_benchmark.md`

## 使用建议与注意事项

### 1. 当前仍是研究型实验平台

当前仓库更适合用于：

- 方法验证
- 消融实验
- 趋势分析
- 报告输出
- ONNX 导出与验证

它目前还不是生产级 TVM pass。

### 2. 区分估计延迟和真实延迟

仓库中当前有两类性能数据：

- 来自硬件感知代价模型的解析式估计延迟
- 来自 ONNX Runtime 的真实基准测试延迟

在写论文表格时，不要把这两类数据直接混在一起，必须明确标注。

### 3. 新实验尽量放到独立目录

如果你后续要新增某一轮论文实验，建议：

1. 在 `experiment_suites/` 下创建一个新的实验目录
2. 将一次性脚本、自定义配置、实验笔记和轻量结果都放进去
3. 只有在可复用算法本身变化时，才修改 `fusion_lab/` 下的核心代码

仓库里已经提供了一个可直接复制的模板目录：

- `experiment_suites/paper_experiment_template/`

## 依赖说明

### Python 环境

当前仓库已经在终端所使用的 Anaconda Python 环境下验证通过。

### 可选本地依赖

一些可选依赖可以安装到 `.vendor/`，并通过 `fusion_lab/vendor.py` 进行加载。
目前这一机制主要用于 `onnxruntime` 等本地依赖。

## 建议阅读顺序

如果你第一次接触这个仓库，建议按下面顺序阅读：

1. `README.md`
2. `PROJECT_LAYOUT.md`
3. `CODEBASE_GUIDE.md`
