# FP32 GPU-resident candidate: rejected

Actual candidate equivalence, fresh restore, benchmark and supervisor exits are all 0; owned child groups have no residual members. The candidate remained explicitly opt-in. Official core source, original smoke and formal main were not modified.

| Input route | Pooled updates/s | GPU util mean / median | Sampled GPU memory peak |
|---|---:|---:|---:|
| Official CPU NumPy FP32 | 348.28 | 49.57% / 49% | 538 MiB |
| Opt-in FP32 GPU-resident gathers | 130.79 | 18.78% / 19% | 540 MiB |

The candidate was **62.45% slower in throughput** and took **2.663 times** as long for the same 6,144 timed updates. Its three windows were 129.60, 129.38 and 133.48 updates/s; all were slower than all baseline windows. Eighty GPU samples fell strictly inside candidate windows, with utilization range 13–23%. Host sample call time grew to 35.8594 of 46.9763 seconds (76.33%). These calls include the added GPU gather dispatch path; the measurement does not isolate CPU arithmetic or prove a disk/H2D bottleneck.

Decision: **do not select this candidate; retain the official CPU NumPy FP32 input path**. Extra gather and host/device indexing dispatch is consistent with the observed overhead, but per-kernel causality was not profiled. No candidate repair or additional optimization matrix is required.

The candidate strictly matched input dtype/shape/bytes, host NumPy RNG, raw loss/gradients and complete 153-leaf state for **two updates**, plus validation, save and independent fresh restore. This numerical equivalence is not claimed for the later 6,208-update diagnostic trajectory, final weights or SR. No uint8 quantization was used.

Sampled peak own RSS sum was 1,543,430,144 bytes (conservative sum, not PSS); max/OOM/OOM-kill event deltas were zero. Both routes used B1024 and official state46/goal46/alpha1.0 on the repeated eight-row artificial-boundary fixture. Full official dataset throughput, formal training and SR remain untested. Exact evidence hashes and window statistics are in `performance_candidate_analysis_v1.json`.
