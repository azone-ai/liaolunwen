# Torch Plan Backend Benchmark

| Method | Mean (ms) | Speedup | Compile (ms) | Search (ms) | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Allclose | Max Abs Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none | 16.5592 | 1.0000 | 1172.53 | 8.244 | 0.927 | 25.13 | 0.000 | False | 0.00560093 |
| dp_paper | 9.2709 | 1.7861 | 297.80 | 61.745 | 0.601 | 52.17 | 3.552 | False | 0.00481625 |
| hw_no_thread_search | 9.2031 | 1.7993 | 297.46 | 36.924 | 0.607 | 51.57 | 3.952 | False | 0.00465155 |
| hw_aware | 8.6331 | 1.9181 | 281.05 | 73.492 | 0.607 | 51.57 | 3.952 | False | 0.00465155 |