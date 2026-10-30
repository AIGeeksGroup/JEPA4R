# Cube downstream on 101732Test

## Separate official state baseline

`official_state_v1/` adds an independent original state46/goal46 entrypoint and B1024 bounded fixture smoke. The later `official_state_workers_v1/` prepares complete official train/val data and compares the original input route, ahead1 background materialization, and2/4 read workers. Prefetch1 is selected for future explicit use after short numerical gates and a23.39% median throughput gain against a direct original-sampler control. Official algorithms and default main remain unchanged. See `official_state_workers_v1/RESULTS.md`, `selected_input.json` and `ROOT_COMPLETION_RECEIPT.json`; formal1M and SR are not started. Long cross-process final-model bitwise equivalence remains unproven, including original A/B controls.

Remote root: `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1`.
User approved Temp storage. No formal downstream training, embedding production or SR is started by preparation.

## Included

- Exact official OGBench/GCIQL source at `source/ogbench`, commit `a977253ce5b88434f4e4f531c50047cc95993a51`, including simulator meshes.
- Actual current LeWM source, its complete Stable World Model submodule and model configuration at `source/lewm_pcd`; RGB/PCD preprocessing and factory are reused.
- Existing GCIQL cached-training, terminal-cache and Native evaluation source closure at `reference/downstream`, preserved as reference. Historical Single-only entrypoints are not active Triple launchers.
- Accepted Triple 2D epoch10 bare checkpoint and configuration at `checkpoints/2d_epoch10`.
- Closed Triple 3D epoch6 bare checkpoint and configuration at `checkpoints/3d_epoch6`. This is an INTERMEDIATE model for interface smoke. Its original training is still active; the accepted final epoch10 model is not present.
- Exact dependency lock, transport/source manifests, logs, actual exits and resource limits. The target base environment is untouched.

## Minimal active interfaces

`cube_inputs.py` handles Cube count and projection only. `cached_inputs.py` delegates sampling, goal selection, index-equality reward and RNG to official `GCDataset`, then maps the same raw IDs to frozen feature rows. `checkpoint_encoder.py` uses the original factory and preprocessors and calls only frozen encoder+projector.

The reused GCIQL implementation already accepts an optional example goal shape. No new loss/update/optimizer implementation is introduced; its non-factory methods are compared to the official source by AST.

| Cube count | Official state | Visual current | Visual goal | Binding |
|---|---:|---:|---:|---|
| 1 | 28 | 211 | 195 | Reserved, weights/data unbound |
| 2 | 37 | 211 | 198 | Reserved, weights/data unbound |
| 3 | 46 | 211 | 201 | Active preparation profile |
| 4 | 55 | 211 | 204 | Reserved, weights/data unbound |

Current input is latent192+proprio19. Goal input is same-ID latent192+each cube's XYZ3; goal arm state is excluded from the state suffix. Goal visuals may contain the arm. Oracle input additionally supports a zero192 prefix+full state. Training consumes offline caches, without encoder calls per batch. Online evaluation must bind actual current/goal scene encoding; the old Native source is not a substitute for a verified Triple binding.

`cube_profiles.json` records paths and official state-GCIQL task settings: alpha1.0/B1024/1M/seed0; reward `(source_idx==value_goal_idx)-1` and mask1-success. Other defaults remain the official config. The active Triple camera is original front+local-X5 degrees, no pullback, approved crop `[[-.25,-.5,-.02],[.75,.5,.9]]`. Do not silently reuse historical Single rewards, horizon200 or state28 constants.

## Tests and status

After dependency installation, the persistent wrapper runs sequentially: CPU count/projection test; one reset/zeroaction and 224RGB/depth render; strict frozen 2D/3D loading and encoder forward on eight actual paired RGB/PCD rows; a B16 diagnostic GCIQL test with two updates and validation; fresh-process full-state reload and finite actions. It retains the official formal B1024 setting but does not claim B1024 memory capacity. The fixture has an explicit artificial end at row7; it is not a complete official dataset or formal SR evidence.

Read `READY.json` together with `finalize_actual_exit_code`; it adopts the actual individual test exits and reports. `migration_acceptance.json` proves the immutable transfer. `deployment_freeze_v4.json` identifies the tested helpers. The old mirror/collector failures remain recorded; no old exit code or PASS field was rewritten.

The real encoder test found an HWC/CHW boundary error; only the new input bridge was corrected. The GCIQL smoke's missing save directory was fixed in an isolated v4 namespace. Its two GPU updates, validation, save and fresh full-state reload each actually passed. The v4 collector then used a different report filename order; `actual_report_filename_alias_receipt.json` binds the genuine unchanged report bytes, and the final CPU-only reader closes acceptance without rerunning NN.

## Manual use

```bash
source /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/activate.sh
python "$CUBE_DOWNSTREAM_ROOT/cube_inputs.py"
```

Activation does not pin the smoke's GPU fraction, thread cap, CPU affinity or PRNG/precision flags. Keep official `impls/utils` and LeWM's `utils.py` imports in separate processes unless the existing Native namespace bridge is explicitly bound and validated.

## Still required before formal experiments

- Accepted final Triple 3D checkpoint from the original WM job, with actual final provenance.
- Full official state train/val data are accepted at `data/official_state/` (3000/300episodes). Aligned rendered/cache assets for visual routes remain unprepared. No1/2/4Cube datasets or models were added.
- Full-cache/model/source identity binding, formal B1024 memory/throughput check, and Triple Native scene/goal/control integration acceptance.
- An explicit formal-training/SR launch instruction. Reserved 1/2/4Cube profiles do not authorize those experiments.
