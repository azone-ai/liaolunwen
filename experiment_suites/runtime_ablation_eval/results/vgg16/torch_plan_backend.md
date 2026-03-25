# Torch Plan Backend Benchmark

| Method | Mean (ms) | Speedup | Compile (ms) | Search (ms) | Avg Occ. | Avg Reg/Thr | Avg SMem/Block (KiB) | Allclose | Max Abs Diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| none | 5.6912 | 1.0000 | 165.63 | 4.194 | 0.773 | 39.35 | 0.000 | False | 0.00217724 |
| dp_paper | 5.1694 | 1.1009 | 112.97 | 20.419 | 0.756 | 40.89 | 0.063 | False | 0.00217724 |
| hw_no_thread_search | 5.1434 | 1.1065 | 115.73 | 12.400 | 0.756 | 40.89 | 0.067 | False | 0.00217724 |
| hw_aware | 5.1000 | 1.1159 | 120.04 | 24.228 | 0.756 | 40.89 | 0.076 | False | 0.00217724 |