# 论文消融表（真实后端）

本表聚焦细粒度消融，口径仍为 `Torch plan backend (trace on CUDA)`。

## ResNet50

| 方法 | Runtime (ms) | Speedup vs none | Search Time (ms) |
| --- | ---: | ---: | ---: |
| dp_paper | 6.2551 | 1.3042 | 105.992 |
| hw_no_thread_search | 9.1405 | 0.8925 | 64.219 |
| hw_no_geometry | 10.7271 | 0.7605 | 129.387 |
| hw_no_register | 9.4179 | 0.8662 | 128.833 |
| hw_no_shared_memory | 9.5676 | 0.8527 | 126.530 |
| hw_no_icache | 9.7532 | 0.8365 | 130.207 |
| hw_aware | 8.8095 | 0.9261 | 84.497 |

## VGG16

| 方法 | Runtime (ms) | Speedup vs none | Search Time (ms) |
| --- | ---: | ---: | ---: |
| dp_paper | 4.9551 | 1.1282 | 28.040 |
| hw_no_thread_search | 5.5987 | 0.9985 | 15.877 |
| hw_no_geometry | 4.9483 | 1.1297 | 28.849 |
| hw_no_register | 4.9494 | 1.1295 | 28.998 |
| hw_no_shared_memory | 4.9767 | 1.1233 | 22.735 |
| hw_no_icache | 5.0703 | 1.1025 | 22.125 |
| hw_aware | 5.0215 | 1.1133 | 22.801 |

## AlexNet

| 方法 | Runtime (ms) | Speedup vs none | Search Time (ms) |
| --- | ---: | ---: | ---: |
| dp_paper | 2.6266 | 1.0589 | 7.832 |
| hw_no_thread_search | 2.5293 | 1.0996 | 5.076 |
| hw_no_geometry | 2.4654 | 1.1281 | 10.425 |
| hw_no_register | 2.4962 | 1.1142 | 9.661 |
| hw_no_shared_memory | 2.4147 | 1.1518 | 10.522 |
| hw_no_icache | 2.4205 | 1.1490 | 9.880 |
| hw_aware | 2.4910 | 1.1165 | 10.195 |

## MobileNetV2

| 方法 | Runtime (ms) | Speedup vs none | Search Time (ms) |
| --- | ---: | ---: | ---: |
| dp_paper | 6.6897 | 1.2800 | 56.113 |
| hw_no_thread_search | 6.6987 | 1.2782 | 32.251 |
| hw_no_geometry | 6.5862 | 1.3001 | 67.088 |
| hw_no_register | 6.5647 | 1.3043 | 65.900 |
| hw_no_shared_memory | 6.7576 | 1.2671 | 67.347 |
| hw_no_icache | 7.5480 | 1.1344 | 65.862 |
| hw_aware | 6.6132 | 1.2948 | 65.560 |

## 结论摘要

- `AlexNet` 和 `MobileNetV2` 上，完整 `hw_aware` 不是最优消融版本，说明当前惩罚权重仍有继续校准空间
- `VGG16` 上，`dp_paper` 与若干去单项惩罚版本略优于 `hw_aware`，说明当前硬件惩罚略保守
- `ResNet50` 上，`geometry` 项影响最大，去掉它反而更慢，说明当前真实后端与解析式几何建模之间还存在偏差
