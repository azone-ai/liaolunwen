# Hardware Analysis: main on generic_gpu

| Method | Feasible | Avg Occ. | Avg Reg/Thr | Max Reg/Thr | Avg SMem/Block (KiB) | Max SMem/Block (KiB) | Avg Threads/Block | Max Threads/Block | Avg Penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | True | 0.927 | 25.13 | 49 | 0.000 | 0.000 | 128.00 | 128 | 1.0000 |
| greedy | True | 0.640 | 49.14 | 72 | 2.501 | 4.500 | 118.17 | 128 | 1.0000 |
| dp_paper | True | 0.601 | 52.17 | 72 | 3.552 | 4.500 | 116.90 | 128 | 1.0000 |
| hw_no_thread_search | True | 0.607 | 51.57 | 72 | 3.952 | 4.500 | 130.57 | 256 | 1.0004 |
| hw_aware | True | 0.607 | 51.57 | 72 | 3.952 | 4.500 | 130.57 | 256 | 1.0004 |

## Notes

- `Avg` metrics are latency-weighted averages across fused blocks in the current plan.
- Shared memory metrics are reported in KiB per block.
- `Avg Penalty` reflects the average soft-penalty multiplier under the current cost model.