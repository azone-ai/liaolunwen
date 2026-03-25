# vgg16 Combined Experiment Results

| Method | Feasible | Est. Latency (ms) | Runtime Mean (ms) | Runtime Speedup | Search Time (ms) | Kernels | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Avg Threads/Block | Allclose | Max Abs Diff |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none | True | 5.0558 | 298.0192 | 1.0000 | 3.659 | 41 | 0.773 | 39.35 | 0.000 | 128.00 | True | 0.00000000 |
| greedy | True | 4.8792 | 287.4028 | 1.0369 | 10.118 | 33 | 0.756 | 40.89 | 0.063 | 127.20 | True | 0.00000000 |
| dp_paper | True | 4.8792 | 281.9216 | 1.0571 | 18.389 | 33 | 0.756 | 40.89 | 0.063 | 127.20 | True | 0.00000000 |
| hw_no_thread_search | True | 4.8853 | 314.4426 | 0.9478 | 11.158 | 33 | 0.756 | 40.89 | 0.067 | 132.04 | True | 0.00000000 |
| hw_aware | True | 4.8853 | 318.7931 | 0.9348 | 21.330 | 33 | 0.756 | 40.89 | 0.076 | 132.29 | True | 0.00000000 |