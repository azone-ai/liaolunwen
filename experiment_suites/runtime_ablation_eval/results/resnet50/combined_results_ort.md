# resnet50 Combined Experiment Results

| Method | Feasible | Est. Latency (ms) | Runtime Mean (ms) | Runtime Speedup | Search Time (ms) | Kernels | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Avg Threads/Block | Allclose | Max Abs Diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none | True | 2.5989 | 14.9785 | 1.0000 | 8.556 | 174 | 0.927 | 25.13 | 0.000 | 128.00 | True | 0.00000000 |
| greedy | True | 1.5906 | 271.4774 | 0.0552 | 46.172 | 62 | 0.640 | 49.14 | 2.501 | 118.17 | True | 0.00000000 |
| dp_paper | True | 1.5695 | 255.8469 | 0.0585 | 60.323 | 55 | 0.601 | 52.17 | 3.552 | 116.90 | True | 0.00000000 |
| hw_no_thread_search | True | 1.5712 | 268.6357 | 0.0558 | 37.814 | 56 | 0.607 | 51.57 | 3.952 | 130.57 | True | 0.00000000 |
| hw_aware | True | 1.5712 | 275.1098 | 0.0544 | 72.945 | 56 | 0.607 | 51.57 | 3.952 | 130.57 | True | 0.00000000 |