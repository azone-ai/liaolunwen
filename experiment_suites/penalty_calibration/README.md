# 惩罚项校准实验

这个实验套件专门用于比较不同惩罚 profile 的效果，不修改核心算法入口。

## 当前支持的 profile

- `baseline`
- `global_soft`
- `cnn_seq`
- `cnn_residual`
- `cnn_light`
- `reduction_tf`
- `auto`

## 推荐实验顺序

1. 先跑 `global_soft`
2. 再跑 `auto`
3. 如果 `auto` 在某类模型上明显更稳，再单独对该类 profile 继续细调

## 运行命令

跑全局软化版本：

```powershell
python experiment_suites/model_generalization_ablation/scripts/run_generalization_experiments.py `
  --manifest experiment_suites/penalty_calibration/configs/global_soft_manifest.json `
  --out-dir outputs/penalty_calibration/global_soft
```

跑自动 profile 版本：

```powershell
python experiment_suites/model_generalization_ablation/scripts/run_generalization_experiments.py `
  --manifest experiment_suites/penalty_calibration/configs/auto_manifest.json `
  --out-dir outputs/penalty_calibration/auto
```
