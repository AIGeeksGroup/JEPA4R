# Official Triple state GCIQL smoke

Target: `ssh 101732Test`, RTX4080 SUPER 32760MiB; cgroup RAM limit66.57GB, CPU quota16 cores. This is a bounded smoke, not the formal1M experiment.

The active route uses the unchanged official `agents/gciql.py`, `Dataset/GCDataset` and save/restore functions. Current46/goal46 FP32, action5, B1024, task alpha1.0; official goal sampling, index-equality reward, Adam3e-4,512x3/LN, discount.99, tau.005 and expectile.9 remain. There is no zero prefix, visual mask, encoder, dtype quantization or change to the learning code. The independent formal wrapper previews by default and requires explicitly existing official train/val files before `--execute`.

## Verified

- Two B1024 updates, same-fixture validation, checkpoint save and independent fresh reload all exit0; full153 state leaves match, Adam2/network3 and finite actions verified.
- A single opt-in FP32 resident-input candidate preserves two sampled batches byte-for-byte, NumPy RNG, loss/gradients, both updated full153 states and validation; save/fresh reload also pass. These equality statements cover the tested two updates, not all later updates or final models.
- All child stages and parent supervisors actually exit0 with empty owned process groups. No max/OOM/oom_kill events were observed. Exact parameters, timings, source pins and raw GPU/resource samples are retained.

## Performance decision

Both longer measurements use64 warmup+3x2048 timed updates with full agent/info synchronization at each window boundary. CPU affinity16 and library threads4 are fixed for both routes. GPU utilization is sampled externally every approximately0.5s.

| Input route | Window updates/s | Pooled updates/s | Decision |
|---|---|---:|---|
| Official CPU FP32 |352.50/355.41/337.46|348.28|Keep unchanged|
| Opt-in FP32 resident GPU |129.60/129.38/133.48|about130.8|Reject; about62.4% slower|

The official route's longer window GPU utilization averaged49.6% across30 samples. Host sampling occupied22.9% of wall time, which can overlap GPU execution and is not a direct GPU-idle or storage-read percentage. The candidate issues several extra gather/dispatch operations; it does not improve end-to-end speed. It remains diagnostic-only and is not imported by the default official launcher. No further optimization matrix is run.

## Limits and pending work

The target has only eight real migrated source observations with an artificial terminal boundary. All smoke and timing results repeat this fixture; they do not establish full official dataset integrity, full-data throughput, SR or final model equality. Complete official train/val data must be transferred or downloaded and bound before formal training. No formal1M training or SR was started. State values are continuous FP32, so uint8 image transport does not apply without changing their values.

`closed_state_smoke_evidence_v1.tar.gz` contains only closed small sources/configs/logs/reports/exits. Diagnostic checkpoints remain on the remote target with SHA/size pointers; they are not described as locally backed up models.
