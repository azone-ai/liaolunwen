# Example PowerShell script for a paper-specific experiment.
# Copy and edit this script instead of modifying the main framework workflow.

python -m fusion_lab.cli `
  --onnx-model model/resnet50-v2-7.onnx `
  --hardware configs/hardware/generic_gpu.json `
  --out-dir outputs/template_resnet50 `
  --methods none dp_paper hw_aware `
  --export-method hw_aware `
  --verify-numerical `
  --verification-backend onnxruntime `
  --benchmark-onnxruntime `
  --benchmark-warmup 10 `
  --benchmark-repeat 50
