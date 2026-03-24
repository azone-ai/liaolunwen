# 代码仓库导览

## 总体说明

当前仓库已经按职责拆成三层：

1. `core`：可复用的融合算法核心，尽量保持稳定。
2. `workflow`：把输入、搜索、报告输出、ONNX 导出和验证串起来的流程层。
3. `research`：只服务于实验、验证和论文分析的辅助代码。

如果后续要新增某一轮论文实验，优先在 `experiment_suites/` 下新建独立目录，而不是直接改核心文件。只有当算法本身真的变化时，才建议改 `fusion_lab/` 里的核心逻辑。

## 代码是怎么跑起来的

### 主执行路径

1. `run_experiment.py`
   - 本地便捷启动脚本。
   - 如果命令行传入参数，则直接转发给 `fusion_lab.cli`。
   - 如果没有参数，则运行内置的本地 smoke test 配置。
2. `fusion_lab/cli.py`
   - 只负责命令行参数解析。
   - 决定当前是 JSON 图流程、ONNX 流程还是全样例流程。
3. `fusion_lab/workflow/pipeline.py`
   - 调用实际的流程函数。
   - 把核心搜索、报告生成、ONNX 导出和可选运行时验证串起来。
4. `fusion_lab/research/experiment_runner.py`
   - 执行 `none`、`greedy`、`dp_paper`、`hw_aware` 等方法。
   - 生成实验汇总报告。
5. 核心算法文件
   - `graph.py`、`ops.py`、`fusion_rules.py`、`cost_model.py`、`search.py`、`hardware.py`
   - 这些文件实现真正可复用的融合逻辑。
6. 可选运行时验证
   - `fusion_lab/research/runtime_validation.py`
   - 负责生成随机输入、比较输出一致性、测试 ONNX Runtime 真实延迟。

### ONNX 工作流路径

1. `fusion_lab/onnx_bridge.py` 先把 ONNX 模型导入内部图表示。
2. `fusion_lab/research/experiment_runner.py` 执行指定融合方法。
3. `fusion_lab/onnx_bridge.py` 把目标融合计划重新导出为 ONNX。
4. `fusion_lab/research/runtime_validation.py` 按需做数值一致性验证和 ONNX Runtime 基准测试。

## 文件分类清单

### A. 核心算法文件

| 路径 | 类别 | 主要职责 | 什么时候改 | 什么时候尽量别改 |
| --- | --- | --- | --- | --- |
| `fusion_lab/enums.py` | 核心 | 共享的映射类型与融合决策枚举 | 新增新的语义类别或决策类型时 | 你只是想改报告或实验脚本时 |
| `fusion_lab/graph.py` | 核心 | 内部图结构、拓扑辅助函数、区间切分、边界分析 | 图结构定义或遍历逻辑变化时 | 你只是想加新的 benchmark 指标时 |
| `fusion_lab/ops.py` | 核心 | 算子模式分类与算子级代价估计 | 新增算子支持或改进单算子估计时 | 你只是改流程或 CLI 行为时 |
| `fusion_lab/fusion_rules.py` | 核心 | 融合合法性规则与主导模式规则 | 融合合法性假设变化时 | 你只是调整硬件惩罚项时 |
| `fusion_lab/cost_model.py` | 核心 | 资源估计、roofline 延迟和硬件惩罚 | 硬件感知方法本身变化时 | 你只是新增一种输出文件格式时 |
| `fusion_lab/search.py` | 核心 | baseline、贪心搜索、DP 搜索和结果结构 | 搜索策略或计划数据结构变化时 | 你只是增加运行时 benchmark 时 |
| `fusion_lab/hardware.py` | 核心 | 硬件配置定义与默认值 | 新增硬件参数或新平台时 | 你只是改 ONNX 导入导出时 |
| `fusion_lab/vendor.py` | 核心支撑 | 从 `.vendor/` 加载可选依赖 | 需要接入新的可选依赖时 | 你只是改算法逻辑时 |
| `fusion_lab/core/__init__.py` | 核心导出 | 为核心类型提供统一导出入口 | 想让导入更清晰时 | 正在改算法细节时 |

### B. 流程与入口文件

| 路径 | 类别 | 主要职责 | 什么时候改 | 什么时候尽量别改 |
| --- | --- | --- | --- | --- |
| `fusion_lab/cli.py` | 流程 | 解析参数并分发到不同工作流 | 需要新增 CLI 选项或新运行模式时 | 你只是改融合逻辑时 |
| `fusion_lab/workflow/pipeline.py` | 流程 | 串联图流程、ONNX 流程和验证流程 | 整体运行顺序变化时 | 你只是改单个算子估计时 |
| `fusion_lab/__main__.py` | 入口 | 包级别执行入口 | 很少改，除非包执行方式需要变化 | 大多数情况下都不需要改 |
| `run_experiment.py` | 入口 | 本地便捷启动与 smoke test 配置 | 想优化默认本地运行体验时 | 你是在加可复用实验功能时 |
| `fusion_lab/__init__.py` | 包导出 | 顶层包说明与导出接口 | 想整理顶层导出时 | 你是在改运行行为时 |

### C. 实验与验证文件

| 路径 | 类别 | 主要职责 | 什么时候改 | 什么时候尽量别改 |
| --- | --- | --- | --- | --- |
| `fusion_lab/research/experiment_runner.py` | 实验 | 执行多种方法并写出 JSON、CSV、Markdown 汇总 | 要增加实验指标或新的报告列时 | 你是在改核心搜索语义时 |
| `fusion_lab/research/runtime_validation.py` | 实验 | 生成输入、验证输出、测试 ONNX Runtime 延迟 | 要增加新的 runtime 后端或验证指标时 | 你是在改融合合法性时 |
| `fusion_lab/experiments.py` | 兼容层 | 保持旧导入路径仍可用 | 很少改 | 新逻辑尽量写到 `research/` 里 |
| `fusion_lab/runtime_benchmark.py` | 兼容层 | 保持旧运行时验证导入路径仍可用 | 很少改 | 新逻辑尽量写到 `research/` 里 |

### D. ONNX 桥接文件

| 路径 | 类别 | 主要职责 | 什么时候改 | 什么时候尽量别改 |
| --- | --- | --- | --- | --- |
| `fusion_lab/onnx_bridge.py` | 输入输出桥接 | 导入 ONNX 图并导出融合后的 ONNX 图 | 需要新增 ONNX 算子支持或导出语义时 | 你只是改报告或硬件配置时 |

### E. 测试文件

| 路径 | 类别 | 主要职责 | 什么时候更新 |
| --- | --- | --- | --- |
| `tests/test_pipeline.py` | 核心回归 | 检查映射、合法性剪枝和搜索基本正确性 | 核心逻辑变化时 |
| `tests/test_onnx_bridge.py` | 桥接回归 | 检查 ONNX 导入导出结构 | ONNX 桥接逻辑变化时 |
| `tests/test_runtime_benchmark.py` | 验证回归 | 检查输入生成、数值验证和 ORT benchmark 辅助逻辑 | 运行时验证逻辑变化时 |

### F. 配置、模型与输入输出文件

| 路径 | 类别 | 主要职责 | 说明 |
| --- | --- | --- | --- |
| `configs/graphs/*.json` | 输入配置 | 受控实验用的样例图 | 新增样例结构时放这里 |
| `configs/hardware/*.json` | 输入配置 | 硬件配置文件 | 新增目标设备时放这里 |
| `model/*.onnx` | 输入模型 | 实验中使用的真实 ONNX 模型 | 大文件通过 Git LFS 管理 |
| `experiment_suites/` | 未来实验区 | 一次性实验脚本、配置和笔记 | 论文实验优先放这里 |
| `outputs/` | 生成输出 | 报告、导出模型、benchmark 结果 | 已忽略，不纳入 git |
| `.vendor/` | 本地依赖 | 本地安装的可选依赖，例如 `onnxruntime` | 已忽略，不纳入 git |

## 新需求应该改哪里

### 如果你想新增内容

- 新增一种算子类型：
  - 先改 `fusion_lab/ops.py`
  - 如果合法性也受影响，再改 `fusion_lab/fusion_rules.py`
- 增加新的硬件惩罚项：
  - 先改 `fusion_lab/cost_model.py`
  - 如果需要新硬件字段，再改 `fusion_lab/hardware.py`
- 增加新的搜索策略：
  - 改 `fusion_lab/search.py`
  - 然后在 `fusion_lab/research/experiment_runner.py` 注册
  - 如果要暴露到命令行，再补 `fusion_lab/cli.py`
- 增加新的报告列：
  - 通常改 `fusion_lab/research/experiment_runner.py`
  - 除非数据来源真的在核心搜索阶段，否则不要先去动核心文件
- 增加新的运行时验证后端：
  - 改 `fusion_lab/research/runtime_validation.py`
  - 保持 `fusion_lab/cli.py` 尽量薄，只做参数解析
- 增加新的论文实验：
  - 在 `experiment_suites/<experiment_name>/` 下创建目录
  - 把脚本、自定义配置和笔记放进去
  - 复用 `fusion_lab/` 里的能力，而不是复制核心逻辑

## 常用运行命令

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

### 3. 运行 ONNX 模型并导出融合 ONNX

```bash
python -m fusion_lab.cli ^
  --onnx-model model/resnet50-v2-7.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/resnet50_run ^
  --methods none hw_aware ^
  --export-method hw_aware
```

### 4. 运行 ONNX 数值一致性验证

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

### 5. 运行 ONNX Runtime 延迟基准测试

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

### 6. 运行全部测试

```bash
python -m pytest tests
```

## 你应该看到哪些输出文件

### 搜索与报告输出

- `summary.json`
- `summary.csv`
- `summary.md`
- `<method>_plan.json`
- `latency_comparison.png`，在启用 `--plots` 时生成

### 运行时验证输出

- `numerical_verification.json`
- `numerical_verification.md`
- `onnxruntime_benchmark.json`
- `onnxruntime_benchmark.md`
- `<method>_fused.onnx`

## 后续维护时的安全修改原则

- 可复用算法改动尽量留在 `fusion_lab/` 核心文件里。
- 一次性实验代码尽量放到 `experiment_suites/`。
- 保持 `fusion_lab/cli.py` 足够薄，把真实逻辑放到 `workflow/` 或 `research/`。
- 如果改动只影响报告，不要优先去动 `cost_model.py` 或 `search.py`。
- 如果改动只影响 ONNX 导入导出，不要优先去动融合搜索逻辑。
- 只要核心逻辑或 ONNX 导出语义变化，就尽量补测试。
