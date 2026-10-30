# Official state fixture: longer performance windows

The profile worker, supervisor and owned child group all completed with actual exit 0 and no residual members. It reused the official GCIQL agent and sampler, state46/goal46, B1024, alpha1.0 and FP32 inputs. The eight-row artificial-boundary fixture remains a diagnostic, not a complete official dataset or SR result.

| Window | Updates/s | CPU sample ms/update | CPU sample wall share | GPU samples | GPU util mean / median / range |
|---|---:|---:|---:|---:|---|
| 1 | 352.50 | 0.6508 | 22.94% | 10 | 50.9% / 49% / 44–64% |
| 2 | 355.41 | 0.6438 | 22.88% | 10 | 52.7% / 49.5% / 48–59% |
| 3 | 337.46 | 0.6782 | 22.89% | 10 | 45.1% / 48% / 32–53% |

The pooled 6,144 timed updates took 17.6410 seconds: **348.28 updates/s**. Thirty external GPU readings fall strictly inside the recorded windows: mean **49.57%**, median **49%**, range **32–64%**. Each window synchronizes the complete agent and info trees at its end. External GPU utilization averages a preceding NVML interval, so it is a sampled diagnostic rather than a per-kernel occupancy measure.

CPU sampling consumed 4.0401 seconds of host wall time (22.90%). Host sampling and asynchronous GPU work may overlap; this fraction does not establish the GPU idle cost, nor distinguish CPU sampling, H2D, kernel dispatch and small-network workload limits. The GPU was not saturated in this fixture, but an input-read or H2D bottleneck is **not confirmed**.

Sampled peak own RSS sum was 1,361,412,096 bytes (conservative sum, not PSS). Sampled GPU memory peaked at 538 MiB. Cgroup max/OOM/OOM-kill event deltas were all zero.

Continuous state46 must remain FP32; uint8 would quantize the official inputs. A single default-off candidate could retain the exact NumPy index/goal/RNG stream while gathering from a GPU-resident FP32 table. Prefer one fused gather per batch. Extra GPU gather and host dispatch can outweigh the saved sub-MB transfer, so any improvement requires A/B evidence; none is claimed here. Full-data throughput remains untested because the official dataset is not present on this target.

The earlier under-one-second windows remain in `performance_analysis_v1.json` only as reference. Exact current evidence hashes, exit records and window timestamps are in `performance_profile_analysis_v1.json`. No source changes, GPU launches or downloads were performed by the independent analysis task.
