# RH56 minimal integration plan

> For agentic workers: use subagent-driven-development with one owner per file, freeze before independent review, then verify before local merge.

Goal: integrate the completed native RH56 Relocate experiment as an explicit optional adapter, while leaving existing Adroit learning and execution defaults unchanged.

Architecture: new `lewm_pcd/rh56_adapter` modules delegate to the existing world-model entry and official CortexBench BC loop. One lazy public CLI selects a stage and, for learning stages, an explicit RGB/PCD route. Expert assets, weights, datasets, licenses and environments remain external.

## Frozen scope

- Base/upstream: `09f020747b22176cb145e84845600f9ca8e91efe`, freshly fetched from `origin/cortexbench-adroit`.
- Isolated branch: `codex/rh56-minimal-integration`; original clean checkout and unrelated dirty checkout are retained.
- Backup refs: `codex/backup-rh56-integration-base-20261025` and `codex/backup-cortexbench-adroit-pre-rh56-20261025`.
- Existing learning sources and old entrypoints must be byte-identical to the base. Only the existing root README may gain an opt-in documentation link.
- User authorized local integration/merge, not GitHub push, retraining, new collection or new SR evaluation.
- Native contract: expert observation56, proprio13, action12, 100Hz, max200 steps, original physics/expert; paired RGB256/MSAA8 and PCD1200/NoAA from the same approved camera.
- Whole-episode split and train-only normalization remain RH56-specific. The image uint8/GPU optimization stays default-off and explicitly experimental; its historical model-equivalence failure remains recorded.
- Historical results keep their actual original producer/source hashes. This packaging commit is not described as having repeated the full training experiment.

## Ordered work

1. Data owner adds portable native/camera/collection, direct batch validation/VDS publication and official-task-initial-state adapter. Reuse existing loops and formulas; remove task deployment/cache orchestration.
2. Learning owner adds portable WM/BC bridge and image hook. Reuse original `train.run` and `bc_pvr_train_loop`, model configs/loss/optimizers. Make expert, baseline and EAI paths explicit/checkout-relative.
3. Root adds `scripts/run_rh56.py`: no stage means help only; selected stage is explicit opt-in; `--route rgb|pcd` is required for WM/BC. Help/dry-run do not import simulator/learning dependencies. Collection remains paired with the approved camera default. Add interface tests first, then the thin dispatcher.
4. Root adds examples, external dependency contract, sanitized original result/provenance record and README link. No data/checkpoints/PIDs/Nix paths/runtime snapshots are committed.
5. Freeze all additions. Independently review scope and real behavior boundaries. Verify new unit/interface tests, relevant existing Adroit release regressions, source-file identity and `git diff --check`. Use an existing configured runtime for dependency-sensitive checks if needed; no full retraining/evaluation.
6. Commit and fast-forward the clean repository's local `cortexbench-adroit` branch. Preserve backup refs and isolated checkout. Produce a reviewable diff and validation record; do not push or alter the unrelated dirty repository.

## Success criteria

- New explicit RH56 stages dispatch to reused code with documented external artifacts and route switches.
- Defaults do not load RH56 or mutate existing Adroit behavior.
- Source/math/data contracts and failed equivalence evidence remain accurately labelled.
- Required checks pass or concrete environmental limitations are reported without inventing PASS.
- Local target branch contains the reviewed change, with original work and old results preserved.
