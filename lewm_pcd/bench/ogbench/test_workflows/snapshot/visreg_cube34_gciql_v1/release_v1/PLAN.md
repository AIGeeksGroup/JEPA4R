# VISReg Cube3/4 local candidate plan

Goal: prepare a new serial Cube3 gates -> Cube3 fresh1M/Native250 -> Cube4 gates -> Cube4 fresh1M/Native250 release. Only new auxiliary files/configurations/output roots are written. Original models, source, eight configurations, data, caches, checkpoints and failed attempts remain immutable.

The approved design reuses foundation functional make_config/commands, the frozen original five gate/four formal commands and environments, gpu_lease, and serial's existing PGID/RSS60GiB/new-OOM supervisor. A shared stage runner validates metadata and calls that supervisor; the chain only sequences four independent stage runtimes. Pilot uses the original full accepted data/cache and purpose=pilot; it performs two updates and one cap4 diagnostic episode. Formal uses fresh seed0/B1024/1M and final Native250/cap1000. Only configuration execution_root/output/commands differ from original make_config output. Copy/backup is advisory and has no phase or model hard gate.

- [x] Write metadata positive/negative fixtures first; record their expected RED result.
- [x] Implement original config/command derivation, immutable source/cache/weight identity checks, predecessor report/exit gates, and one-count storage admission.
- [x] Add thin prepare/publish, shared stage runner, four-stage chain/dispatch and read-only progress.
- [x] Run all metadata fixtures, compile all sources without importing NN packages, inspect source/field diff, and record GREEN evidence.
- [x] Prepare exact local source/expected-binding freeze and HANDOFF. Root independently reviews, publishes, verifies remote metadata, then starts if all gates hold. This author performs no SSH/NN or original-file writes.

Storage contract: initial artifact allocation 3,100,000,000 bytes plus 536,870,912 free reserve. Each stage requires free >= max(artifact_budget - bytes_already_paid_in_owned_STORE, 0) + reserve. STORE bytes are deducted once. Pilot checkpoint stat increases the artifact budget if 26 projected checkpoint payloads plus 400MB overhead exceed the initial allocation; no old-data deletion, precision/seed/batch changes or automatic restart.
