# 备份说明

本仓库用于保存“硬件约束感知的 ONNX 算子融合实验”代码、配置、测试脚本以及示例模型。

## 当前备份范围

- 源代码与脚本：`fusion_lab/`、`run_experiment.py`
- 配置文件：`configs/`
- 测试代码：`tests/`
- 文档：`README.md`、本文件
- ONNX 模型：`model/*.onnx`

## 大文件策略

`model/` 目录下的 ONNX 模型通过 Git LFS 管理。

仓库中已经配置：

```gitattributes
model/*.onnx filter=lfs diff=lfs merge=lfs -text
```

这意味着：

- `model/` 下新增的 `.onnx` 文件会按 LFS 跟踪
- 推送前需要本机安装并启用 Git LFS
- 克隆后需要拉取 LFS 对象，才能拿到完整模型文件

## 不纳入备份的内容

以下内容默认不会提交到 GitHub：

- 实验输出：`outputs/`
- 本地 IDE 配置：`.idea/`、`.vscode/`
- 本地缓存：`__pycache__/`、`.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`
- 本地依赖缓存：`.vendor/`
- 临时文件与日志：`tmp/`、`temp/`、`*.tmp`、`*.log`

## 首次在新机器恢复仓库

```bash
git lfs install
git clone git@github.com:azone-ai/liaolunwen.git
cd liaolunwen
git lfs pull
```

如果只想恢复 `main` 分支，执行上述命令即可。

## 日常备份建议

推荐每次完成一个阶段性修改后执行：

```bash
git add .
git commit -m "描述本次修改"
git push origin main
```

## 新增模型时的注意事项

如果新增模型文件并放在 `model/` 目录下，一般不需要额外配置，直接提交即可，因为 `.gitattributes` 已经覆盖 `model/*.onnx`。

如果以后有新的大文件目录，例如 `datasets/` 或 `checkpoints/`，建议单独补充 LFS 规则，而不是直接按普通 Git 提交。
