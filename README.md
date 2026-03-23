# 面向 TVM 算子融合的硬件感知实验框架

这个仓库实现的是一套**可独立运行的实验框架**，用于验证“算子映射分类 + 融合规则 + 动态规划搜索 + 硬件约束惩罚项”这条技术路线。

由于当前环境里没有安装 TVM，本实现先把论文和方案里的核心思想抽象成了**解析式实验平台**：

- 用计算图 JSON 表示网络
- 用映射类型判断融合合法性
- 用硬件画像评估融合后的代价
- 用动态规划寻找全局最优分块方案

后续如果你要接 TVM/Relay，只需要把 Relay 图导出成这里的 JSON 格式，或者补一层 Relay importer 就可以复用搜索与代价模型。

## 对原方案的修正

结合你的 `方案.md` 和 `开题资料.md`，这里对公式和规则做了 4 个关键修正。

### 1. 映射类型补全

你的原方案里有：

- One-to-One
- One-to-Many
- Many-to-Many
- Reorganize
- Shuffle

这里额外补了 **Reduction（多对一）**。原因是 `ReduceSum`、`LayerNorm` 这类算子在 TVM/深度学习图里非常常见，如果不单独建模，会把“规约”和“卷积/矩阵乘”混在一起，规则容易失真。

### 2. 硬约束和软约束分离

原方案里“硬件约束惩罚项”如果直接加到一个统一得分里，量纲会不统一。这里改成：

- **硬约束**：超限直接判定为不可融合
- **软约束**：不超物理上限，但接近上限时用惩罚项放大执行时间

具体做法是：

- `Registers_per_thread > R_thread_max` 时，融合块不可行
- `SharedMem_per_block > S_block_max` 时，融合块不可行
- `Threads_per_block > T_block_max` 时，融合块不可行

### 3. 代价函数改成“时间”而不是任意分数

这里把总代价定义成**估计执行时间**，单位统一为毫秒：

```text
T_base(B, h) = max(F(B) / (eta_c(B, h) * PeakFLOPS(h)),
                   M(B) / (eta_m(B, h) * Bandwidth(h)))
               + T_launch(h)
```

其中：

- `B` 是一个融合块
- `h` 是目标硬件
- `F(B)` 是融合块 FLOPs
- `M(B)` 是融合块外部输入/输出和权重的访存量
- `eta_c`、`eta_m` 会被 occupancy 影响

然后把你的硬件惩罚项写成乘性放大项：

```text
Penalty(B, h) = 1
              + lambda_reg * phi(Reg(B) / R_soft(h))
              + lambda_smem * phi(SMem(B) / S_soft(h))
              + lambda_geom * P_geom(B)
              + lambda_icache * phi(Inst(B) / I_soft(h))
```

其中：

```text
phi(x) = max(0, x - 1)^2
```

最终执行时间：

```text
T_exec(B, h) = T_base(B, h) * Penalty(B, h)
```

这样做的好处是：

- 硬约束和软约束语义清楚
- 所有惩罚项都作用在“时间”上，量纲一致
- 很容易做消融：把软惩罚权重设成 0，就是“论文原版/无硬件惩罚”基线

### 4. 从局部判断改成全局优化

如果只用：

```text
Cost(Fused) < Cost(OpA) + Cost(OpB)
```

去做两两判断，容易陷入局部最优。这里改成对整图做分块优化：

```text
pi* = argmin_{pi in P(G)} sum_{B in pi} T_exec(B, h)
```

其中：

- `G` 是整张计算图
- `P(G)` 是所有合法融合划分集合
- `pi` 是一种划分方式

实现上采用**最长路径分层 + 连续层动态规划**：

```text
dp[i] = min_j (dp[j - 1] + C(L_j ... L_i))
```

这里 `C(L_j ... L_i)` 表示把第 `j` 到第 `i` 层视作一个候选融合区间时的总代价。

## 当前实现包含的模块

- `fusion_lab/graph.py`
  - 计算图数据结构
  - DAG 拓扑序
  - 最长路径分层
  - 区间内弱连通分量划分

- `fusion_lab/ops.py`
  - 算子到映射类型的分类
  - FLOPs / 权重字节数 / 寄存器 / 指令数 / 线程配置估计

- `fusion_lab/fusion_rules.py`
  - 融合合法性矩阵
  - 融合后主导类型推断

- `fusion_lab/cost_model.py`
  - 硬约束剪枝
  - occupancy 估计
  - roofline 风格基准时间
  - 软约束惩罚项

- `fusion_lab/search.py`
  - 单层基线
  - 贪心融合基线
  - 动态规划搜索

- `fusion_lab/experiments.py`
  - 运行不同方法
  - 导出 JSON / CSV / Markdown / 图片结果

## 已提供的实验样例

- `configs/graphs/residual_block.json`
- `configs/graphs/mobilenet_bottleneck.json`
- `configs/graphs/transformer_ffn.json`
- `configs/hardware/generic_gpu.json`
- `configs/hardware/edge_gpu.json`

## 运行方式

### 运行单个样例

```bash
python run_experiment.py ^
  --graph configs/graphs/residual_block.json ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/residual_block ^
  --plots
```

### 一次跑所有样例

```bash
python run_experiment.py ^
  --all-samples ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs/all_samples ^
  --plots
```

### 可选方法

- `none`：不融合
- `greedy`：贪心融合
- `dp_paper`：无软惩罚的动态规划
- `hw_aware`：带硬件惩罚项的动态规划

例如：

```bash
python run_experiment.py ^
  --graph configs/graphs/transformer_ffn.json ^
  --hardware configs/hardware/edge_gpu.json ^
  --out-dir outputs/ffn_edge ^
  --methods none dp_paper hw_aware
```

## 输出结果

每次运行会生成：

- `summary.json`
- `summary.csv`
- `summary.md`
- `<method>_plan.json`
- `latency_comparison.png`（如果加了 `--plots`）

其中 `<method>_plan.json` 里会记录：

- 每个融合块包含哪些节点
- 每个融合块覆盖哪些层
- 估计延迟
- 主导映射类型
- occupancy
- 寄存器和共享内存估计

## 需要注意的假设

这套代码是**实验框架**，不是 TVM 生产编译 pass，所以这里做了几个受控近似：

- 用最长路径分层近似整图融合搜索空间
- 用解析式指标估计寄存器、指令缓存和共享内存压力
- 用 tile 级内部缓存近似融合块片上暂存量
- 默认 Many-to-Many + Many-to-Many 视为不建议融合

这些假设非常适合做论文实验、消融和趋势验证；如果后面你要做 TVM 原型落地，可以把这里的代价模型嵌进 Relay pass，再把统计量替换成真实的 TIR/LLVM 分析结果。

## ONNX 输入输出

现在可以直接把输入换成 ONNX 计算图，并导出融合后的 ONNX：

```bash
python run_experiment.py ^
  --onnx-model path\to\model.onnx ^
  --hardware configs/hardware/generic_gpu.json ^
  --out-dir outputs\model_run ^
  --methods none hw_aware ^
  --export-method hw_aware
```

默认会在 `out-dir` 下生成 `hw_aware_fused.onnx`。如果你想指定输出文件路径，可以额外传 `--fused-onnx-out path\to\fused.onnx`。

当前 ONNX 导入器优先支持单输出节点的推理图。导出的融合 ONNX 使用 `fusion_lab` 域的本地函数来封装原始子图，所以图上会显示为一个融合节点，同时模型内部仍保留原始语义定义。
