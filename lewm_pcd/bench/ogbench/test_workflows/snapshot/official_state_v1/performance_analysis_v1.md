# Official state smoke: short-window performance

Actual correctness, fresh reload, benchmark and supervisor exits are all 0; each child group has no residual members. This is B1024, state46/goal46 and official Triple alpha1.0, using only the repeated eight-row fixture. It is not a full official dataset benchmark or SR result.

| Window | Updates/s | CPU sample ms/update | CPU sample wall share | Strict-window GPU samples |
|---|---:|---:|---:|---:|
| 1 | 347.02 | 0.6633 | 23.02% | 1 |
| 2 | 349.24 | 0.6717 | 23.46% | 1 |
| 3 | 404.20 | 0.5684 | 22.98% | 1 |

The pooled 768 updates took 2.1041 seconds, or 365.00 updates/s. Every timed window synchronizes all agent and info leaves at its end. CPU sampling can overlap GPU execution, so its wall-time share is not a measured GPU idle fraction or a directly removable cost.

Only three GPU readings fall strictly inside these windows: 50%, 50%, 54% (mean 51.33%, median 50%). Each window lasts under 0.74 seconds. NVML utilization averages its own preceding interval and the external readings arrive about 0.5 seconds apart plus command overhead. These samples are insufficient to diagnose a CPU read bottleneck or select an optimization. One later 59% reading is outside the final window and excluded.

Sampled peak own RSS sum was 1,441,628,160 bytes (a conservative sum, not PSS), and sampled GPU memory peaked at 540 MiB. Cgroup max/OOM/OOM-kill event deltas were all zero.

Continuous FP32 state must not be quantized to uint8. No optimization has been selected. Root authorized one longer benchmark-only profile, reusing the accepted smoke and leaving official source and correctness gates unchanged. Machine-readable evidence hashes and exact timestamps are in `performance_analysis_v1.json`.
