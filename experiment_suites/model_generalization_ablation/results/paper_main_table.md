# 论文主表（真实后端）

口径说明：

- 本表采用 `Torch plan backend (trace on CUDA)` 作为更接近真实 fused kernel 的执行口径
- `Kernel Reduction` 相对 `none` 计算
- `Search Time` 为当前机器上的融合搜索与计划构建时间

| 模型 | none (ms) | greedy (ms / x) | dp_paper (ms / x) | hw_aware (ms / x) | hw_aware Kernel Reduction | hw_aware Search Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 8.1581 | 4.8836 / 1.6705 | 6.2551 / 1.3042 | 8.8095 / 0.9261 | 67.8% | 84.497 |
| VGG16 | 5.5902 | 5.9263 / 0.9433 | 4.9551 / 1.1282 | 5.0215 / 1.1133 | 19.5% | 22.801 |
| AlexNet | 2.7812 | 2.9192 / 0.9527 | 2.6266 / 1.0589 | 2.4910 / 1.1165 | 40.0% | 10.195 |
| MobileNetV2 | 8.5625 | 6.7409 / 1.2702 | 6.6897 / 1.2800 | 6.6132 / 1.2948 | 52.0% | 65.560 |

## 可直接写进正文的结论

- `hw_aware` 在 `AlexNet` 和 `MobileNetV2` 上分别取得 `1.1165x` 和 `1.2948x` 的真实后端加速
- `VGG16` 上 `hw_aware` 仍有 `1.1133x` 收益，但略弱于 `dp_paper`
- `ResNet50` 上这轮真实后端结果里 `greedy` 最优，`hw_aware` 未超过 `none`
- 从更多模型角度看，硬件感知方法在轻量 CNN 和部分经典 CNN 上更稳定，重残差模型上还需要继续校准
