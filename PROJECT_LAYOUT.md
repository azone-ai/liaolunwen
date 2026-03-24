# 项目结构总览

## 一、整体流程代码

- `fusion_lab/workflow/`：端到端流程编排层。
  - `pipeline.py`：负责图流程、ONNX 流程与结果汇总打印。
  - `__init__.py`：工作流导出入口。
- `fusion_lab/cli.py`：轻量命令行入口，只负责解析参数并分发到具体流程。
- `run_experiment.py`：本地便捷启动脚本。
- `fusion_lab/onnx_bridge.py`：供流程层调用的 ONNX 导入导出桥接。

## 二、可复用核心代码

- `fusion_lab/core/`：可复用算法核心的分类导出入口。
- `fusion_lab/enums.py`：映射类型与融合决策枚举。
- `fusion_lab/graph.py`：图结构与拓扑辅助逻辑。
- `fusion_lab/ops.py`：算子元信息与代价估计辅助逻辑。
- `fusion_lab/fusion_rules.py`：融合合法性规则。
- `fusion_lab/cost_model.py`：硬件感知代价模型。
- `fusion_lab/search.py`：贪心与动态规划融合搜索。
- `fusion_lab/hardware.py`：硬件配置抽象。
- `fusion_lab/vendor.py`：本地依赖导入辅助工具。

## 三、实验与验证代码

- `fusion_lab/research/`：实验专用辅助层。
  - `experiment_runner.py`：批量执行方法并生成实验报告。
  - `runtime_validation.py`：ONNX Runtime benchmark 与数值一致性验证。
- `fusion_lab/experiments.py`：面向旧导入路径的兼容包装。
- `fusion_lab/runtime_benchmark.py`：面向旧运行时验证导入路径的兼容包装。
- `tests/`：回归测试与运行时验证测试。

## 四、未来实验的组织约定

- 每一轮新的论文实验，尽量放在 `experiment_suites/<experiment_name>/` 下。
- 可复用算法逻辑尽量保留在 `fusion_lab/` 中。
- 一次性分析脚本、自定义配置、实验笔记和表图草稿，尽量放到对应的 `experiment_suites/<experiment_name>/` 中。
