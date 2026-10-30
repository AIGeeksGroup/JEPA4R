# Complete official Triple state data and ordered input comparison

Target `ssh101732Test` means SSH alias `101732Test`. Hardware: RTX4080SUPER32760MiB, CPU quota16, cgroup RAM limit66.57GB. This work prepares data and runs bounded diagnostics only. No formal1M training, Native evaluation or SR was started.

## Data accepted

Original official files are at `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/data/official_state/`:

| File | Raw rows | Episodes | Bytes | SHA256 |
|---|---:|---:|---:|---|
|cube-triple-play-v0.npz|3,003,000|3000|1,003,544,255|4e09a9f5ba7eaba59350dbef7e85887b9a2746490c69eaa83af34a3f9029202b|
|cube-triple-play-v0-val.npz|300,300|300|100,613,267|bd6cb3817fc464d7afc50bd90d766f353bc5b8d097579c70afe680a9aa0404a6|

Official HTTPS URL/ETag/ranges, full target SHA and ZIP CRC, every raw field schema/finite check, and actual original compact loader semantics passed. Train/val compact fields occupy700,299,600B total. A separate stat-freeze binds accepted file paths, sizes, mtime/ctime/device/inode and the acceptance SHA, avoiding repeated full-file hashes during timing. No raw observations/actions were rewritten. The original slow remote/urllib attempts and retired states remain recorded; the final curl transport, SCP and target validation all have their own genuine successful exits. One retired Windows process has no recovered kernel exit code and is not relabelled successful.

## Numerical checks

All original GCIQL/model/optimizer/sampler files remain byte-for-byte unchanged. State46/goal46/action5 FP32, B1024, alpha1.0 and every other official state-profile hyperparameter remain. No zero prefix, mask, uint8 conversion, GPU-resident table, encoder, new data or model is introduced.

Eight target pytest tests passed with real JAX/Flax, followed by one full-data GPU gate covering baseline/prefetch1/workers2/workers4. The tested train→train→validation→save-boundary→train sequence preserves all batch keys, dtype/bytes and consumed/boundary NumPy RNG. Raw loss/gradients, update info and final153 state leaves match exactly for the tested three updates. One original checkpoint3 save and independent fresh reload pass with Adam3/network4, finite actions, and the next train batch. Host NumPy RNG is stored in an explicit auxiliary file, not falsely attributed to the original agent checkpoint.

## Measured performance and selection

Each timing process uses the same real train/val files, example/seed0,64warmup+3x2048 timed updates, B1024,CPU affinity16/library threads4/JAX default precision, and full agent/info synchronization at each window end. Sampling/wait/dispatch are included. Formal periodic validation, logging, evaluations and end-to-end experiment duration are excluded.

| Route | Median updates/s | Relative to direct original control | GPU mean in measured windows |
|---|---:|---:|---:|
|Direct original sampling control|275.40|reference|not sampled in this final control|
|One background materializer, ahead1|339.81|+23.39%|44.36%|
|Two row readers plus one coordinator|296.26|+7.58%|42.00%|
|Four row readers plus one coordinator|267.22|-2.97%|39.97%|

The thin-interface original controls measured263.62 and274.49 updates/s (median drift4.13%). A final literal control removes the diagnostic baseline's per-batch RNG snapshot and directly calls the original sampler. Prefetch1 exceeds the fastest original control by23.39%, above the predeclared5% adoption threshold. Select `prefetch1` for subsequent explicit use; two/four read-worker modes remain unselected. The default official launcher and adapter baseline default remain unchanged because formal training was not authorized.

Four-role readers retain one coordinator separately; configured row-worker counts were actually observed as2 and4. No multiple trainers or optimizer-update sharding is used. All actual child and parent exits are0, groups/reader threads are drained, with no max/OOM/oom_kill increments. The largest sampled own-process RSS was about3.83GiB, below the reused4GiB guard; this is a sampled RSS measure, not a claim of exact PSS or a full-training guarantee.

## Numerical limits retained

The strict equality claim covers the stated short gate and input/RNG evidence. Across independent6208-update timing processes, final model fingerprints differ, including original baseline A versus B; long-window model bitwise equivalence is unproven. No precision, autotune or tolerance was changed to hide this observation. The speed selection is not evidence that final weights or SR will be identical or improved.

Use `selected_input.json` and the frozen `OrderedInput` API for the prepared setting. A future launcher must enforce the documented barrier before validation/evaluation/save/end; this preparation does not silently install it into the original main. Raw full NPZ files remain on target Temp and dedicated local Temp. Only closed source/config/log/exit/report/golden-batch evidence is backed up to the project; diagnostic checkpoint files remain remote with SHA/size pointers.
