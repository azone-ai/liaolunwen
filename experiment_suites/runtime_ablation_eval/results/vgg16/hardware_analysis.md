# Hardware Analysis: mxnet_converted_model on generic_gpu

| Method | Feasible | Avg Occ. | Avg Reg/Thr | Max Reg/Thr | Avg SMem/Block (KiB) | Max SMem/Block (KiB) | Avg Threads/Block | Max Threads/Block | Avg Penalty |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | True | 0.773 | 39.35 | 42 | 0.000 | 0.000 | 128.00 | 128 | 1.0000 |
| greedy | True | 0.756 | 40.89 | 57 | 0.063 | 4.500 | 127.20 | 128 | 1.0000 |
| dp_paper | True | 0.756 | 40.89 | 57 | 0.063 | 4.500 | 127.20 | 128 | 1.0000 |
| hw_no_thread_search | True | 0.756 | 40.89 | 57 | 0.067 | 4.500 | 132.04 | 256 | 1.0013 |
| hw_aware | True | 0.756 | 40.89 | 57 | 0.076 | 9.000 | 132.29 | 256 | 1.0013 |

## Notes

- `Avg` metrics are latency-weighted averages across fused blocks in the current plan.
- Shared memory metrics are reported in KiB per block.
- `Avg Penalty` reflects the average soft-penalty multiplier under the current cost model.