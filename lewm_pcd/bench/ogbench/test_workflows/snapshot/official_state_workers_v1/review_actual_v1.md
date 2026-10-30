# Independent actual-artifact review

Verified: complete official train/validation data (3000/300 episodes), four-mode same-process three-update numerical gate, real validation, checkpoint3 and fresh agent/NumPy next-batch restore. All actual exits are0, with no residual child processes.

| Input mode | Median updates/s | Timed GPU utilization mean | Peak own RSS GiB |
|---|---:|---:|---:|
| baseline_A | 263.617 | 38.27% | 2.032 |
| prefetch1 | 339.810 | 44.36% | 3.565 |
| workers2 | 296.265 | 42.00% | 2.013 |
| workers4 | 267.223 | 39.97% | 3.828 |
| baseline_B | 274.495 | 41.22% | 2.027 |
| literal original sampler | 275.402 | not sampled | 2.017 |

Selected candidate: prefetch1. Its median throughput exceeds the fastest control by 23.387%. Original control A/B drift is 4.126%.

The two original controls use the same thin input interface, including a caller check and NumPy state snapshot per batch. The literal control calls the original Dataset.sample directly. Every benchmark uses64 warmup plus3x2048 timed updates and synchronizes the whole agent and info at each window tail.

Verified: every window-tail batch/RNG digest and final host NumPy RNG agree across all six timing runs. No max/oom/oom_kill increments were observed. GPU means use only monitor samples inside actual timing windows.

Not proven:6208-update cross-process model equivalence. All independent final153 fingerprints differ, including original A/B controls. The cause is unverified. Same-process three-update raw loss/gradient/info/final153 comparisons remain exactly equal; no tolerance was relaxed.

This selects a performance candidate under the declared short gate. Official defaults remain unchanged. No formal1M training or SR was started.
