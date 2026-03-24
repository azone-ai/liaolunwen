# 实验套件目录

这个目录专门用于存放某一轮论文实验的独立文件夹。

## 推荐规则

- 可复用框架代码继续放在 `fusion_lab/`。
- 一次性实验脚本、实验笔记和自定义配置，放在这里的独立子目录里。

## 已提供的起始模板

仓库已经提供了一个可直接复制的模板目录：

- `experiment_suites/paper_experiment_template/`

## 推荐目录结构

```text
experiment_suites/
  <experiment_name>/
    README.md
    notes.md
    configs/
    scripts/
    results/
    figures/
    tables/
    logs/
```

## 未来实验目录命名示例

- `runtime_vs_estimate`
- `edge_gpu_ablation`
- `resnet50_vgg16_main_table`
- `transformer_scaling_study`
