# 论文实验模板目录

这个目录是新建一轮论文实验时的起始模板。

## 目的

当你希望做一组相对独立的实验时，建议使用这个目录，而不是把一次性代码直接塞进可复用框架中。

典型场景包括：

- 新的消融实验
- 新的硬件对比实验
- 新的模型家族 benchmark
- 新的论文表格或图生成流程

## 推荐使用方式

1. 复制这个目录，或者在 `experiment_suites/` 下新建同级目录，例如 `experiment_suites/edge_gpu_ablation/`。
2. 可复用算法变化继续放在 `fusion_lab/` 中。
3. 这个目录里只放实验专用配置、脚本、笔记和结果汇总。
4. 将本轮实验的精确运行命令记录到 `notes.md` 中，方便复现。
5. 大体积原始输出尽量不要提交到 git。

## 已包含的子目录

- `configs/`：只服务于这一轮实验的配置文件
- `scripts/`：编排本轮实验的脚本
- `results/`：结果汇总或轻量导出表格
- `figures/`：论文图或绘图脚本
- `tables/`：论文可直接使用的表格或 LaTeX 片段
- `logs/`：命令日志、benchmark 记录或调试记录

## 目录命名建议

建议使用带语义的目录名，例如：

- `experiment_suites/runtime_vs_estimate/`
- `experiment_suites/edge_gpu_ablation/`
- `experiment_suites/resnet50_vgg16_main_table/`

## 输出提交建议

建议提交：

- 配置文件
- 脚本
- 实验笔记
- 轻量的 Markdown、CSV 或 LaTeX 输出

通常不建议提交：

- 大体积 profiler 原始 trace
- 大量中间导出的 ONNX 文件
- 体积很大的 benchmark 缓存
