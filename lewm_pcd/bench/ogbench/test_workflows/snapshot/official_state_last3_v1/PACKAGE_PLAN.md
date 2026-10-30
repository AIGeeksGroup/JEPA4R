# Last3 closed SR backup plan

Prepared only; the collector has not been executed. Root must statically review this file and the script, confirm the actual run is closed, and provide a nonempty `--closed-signal` before execution.

Fixed remote case: `/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1` (`D`), with `H=D/official_state_last3_v1`, `R=H/runtime_v2`, `A=H/attempt_v2`. Model directory is read from `R/entry_binding.json` and constrained to the original formal runtime's `model_outputs`. Seed ledgers come only from `D/official_state_formal_v1/runtime`.

## Required closure

- Trainer JSON and actual exit file are 0; `residual_members=[]`; A supervisor actual exit is 0.
- R entry status and final event are `last3_evaluation_returned`, exit 0; supervisor status is completed; runtime check exit 0.
- Exactly 800K/900K/1M, zero new updates, no dataset load, 5 tasks, 50 measured + 1 video episode/task, 1000 steps, CPU, temperature 0, Gaussian null.
- Three successful reports with `RNG_ledger_match=true`, 15 ordered task returns, 3 checkpoint returns; CSV and report metrics agree. Five per-task SRs use denominator 50; overall SR is their mean. Total: 750 measured + 15 video episodes. Video is excluded from SR by the pinned original evaluation source.
- Four successful restore receipts: each selected checkpoint plus the 1M template; 153 finite state leaves, Adam count = epoch, network step = epoch + 1. Receipts are preserved evidence; the collector does not restore or deserialize a model.

## Exact allowlist

The Python constants `R_NAMES`, `A_NAMES`, `H_NAMES`, `H_OPTIONAL`, `DEPENDENCIES`, and `MIGRATION_PINS` plus the selected3 exact model/ledger filenames are the complete fixed allowlist. There is no recursive scan. Required entries must exist; the explicitly named optional Root-local metadata and v1 missing entry status may be absent on target. Symlinks and nonregular sources fail. Small files are capped at 32 MiB; final model at 64 MiB.

- R: small entry/binding/events/status/argv/plan/freeze/runtime/resource/train logs, actual exit files, CSV, AST proofs, old-CSV protection, selected3 reports and four restore receipts.
- A: plan, freeze, dispatch receipt/log, supervisor actual exit.
- H: collector/plan, original and v2 entry/supervisor/run sources, tests, freezes, reviews, CPU/config-prefix evidence; original v1 failure status/exits/logs. Root-local start/dispatch snapshots are optional by exact name only.
- D: exact original main/evaluation/GCIQL, related utilities, frozen formal/guard/input dependencies and dependency lock. These are compared to A's historical launch freeze. `encoders.py` (4644 bytes, SHA `6bd7fdd82838b754d26af01f9e84d70fa63fca1e09efee38a09f325884073ecc`) and `networks.py` (16753 bytes, SHA `e7e4c4a439b691911e5ae6c2653f253c5703e6b5d190d813e28ebc32e7306094`) use original migration manifest entries read from the local coordination `manifest.json` during preparation.
- Original model: `flags.json`, selected3 original checkpoint sidecars and small NumPy RNG sidecars; original formal `entry_binding.json`, selected3 original seed ledgers and before/after NumPy RNG NPZs. These NPZs are small RNG records, not raw data.
- **Only `params_1000000.pkl` is copied.** All selected3 pickle sizes/stats are checked against original sidecar metadata. 800K/900K SHA and size are copied as pointers from original sidecars, bound to restore receipts; their pickle contents are never opened or rehashed. The archived 1M bytes must match the old sidecar SHA.

Original evaluation source: `source/ogbench/impls/utils/evaluation.py`, SHA256 `4d9a2732bc9ef89ce99fbc0f07a76d3f4d035783af3a658251709140ff9de3d5`. The original actor seed ledgers are retained; prior live environment initial scenes are not claimed to be reproduced.

Excluded: raw datasets and all raw NPZ reads/hashes, preprocessed cache, environment/runtime library trees, tmp, WandB, media/video, other checkpoint pickle contents, old large evidence archives. No source changes, new training/evaluation, deletion, overwrite, or cleanup occurs.

## Output and integrity

Output must be fresh `H/closed_evidence_v1`. Existing output fails without modification. The collector uses nice increment 10 and at most the first two allowed CPUs. After closed gates pass, it creates closure checks, checkpoint pointers, package policy, one tar.gz, an external member manifest, and a final receipt. The tar includes `_backup/evidence_manifest.json`, so only the archive needs one network copy; the manifest describes every member except itself, and the receipt binds its bytes by SHA. Every archived source has stats before/after reading and after archive verification; all tar member lengths and SHA256 values are checked; the full gzip stream is consumed through EOF/CRC; archive SHA256 is recorded. Builder stdout returns the receipt JSON with archive size/SHA and embedded manifest SHA/name. The successful receipt is written only after those checks pass. Failures retain their partial output for Root's inspection and do not write a successful receipt. The receipt's `actual_exit_code=0` means archive verification completed; Root still separately records the real builder process exit and SCP exit, and verifies the local downloaded bytes.

Exact target invocation after Root review and closure signal:

```sh
/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/env_runtime/bin/python -B /root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1/official_state_last3_v1/build_last3_evidence.py --closed-signal ROOT_CONFIRMED_LAST3_CLOSED
```

Do not use this placeholder as a closure decision. Root replaces it with the concrete recorded closed-run signal and captures stdout, stderr and actual process exit separately. No execution, tests, archive, weight copy/load or remote login was performed to prepare these two files.
