# RH56 packaging validation, 2026-10-25

Base/upstream: `09f020747b22176cb145e84845600f9ca8e91efe`.

Reviewed executable candidate tree: `4d440ff6372f097cf92ba9ce67e36fad44d2df73`.
Candidate archive SHA256: `d2cc652a8ff59826baa4bfda6fb75a7247e893ff39b30c39d50b09eba9411745`.

The final documentation-only addition of this validation record does not change the tested adapter, entry, model or test sources. Existing shared learning sources and Adroit entrypoints are unchanged from the base. The sole existing-file change is the README opt-in link.

## Local checks

Python3.12.14, NumPy2.3.5, pytest8.4.2 and h5py3.14.0. Test-only packages were isolated outside the repository and did not replace a training environment.

```text
python -B -m pytest -p no:cacheprovider -q \
  lewm_pcd/scripts/checks/test_rh56_entry.py \
  lewm_pcd/scripts/checks/test_rh56_training_bridge.py \
  lewm_pcd/scripts/checks/test_rh56_data_adapter.py \
  lewm_pcd/scripts/checks/test_adroit_release_scope_20261006.py \
  lewm_pcd/scripts/checks/test_adroit_release_loader_20261006.py \
  lewm_pcd/scripts/checks/test_adroit_rgb_launcher.py
```

Result: **42 passed, 2 skipped**. Both direct-batch HDF/VDS tests executed and passed. The skipped cases are optional historical Pen real-checkpoint CPU parity tests; their required reference loader/checkpoints/runtime were not configured locally. They are not reported as passing.

The public entry tests first failed because the entry did not exist, then passed after implementation. Coverage includes opt-in/no-op defaults, dependency-free help/dry-run, explicit modality, backend argument forwarding, approved paired-camera default, missing external inputs, environment restoration and data-stage delegation.

## Existing Linux runtime

Python3.10.21, PyTorch2.8.0+cu128, CUDA build12.8, NumPy1.26.4, h5py3.16.0, MuJoCo3.12.0 and stable_pretraining0.1.8.

```text
python -B -m unittest discover -s lewm_pcd/scripts/checks -p 'test_rh56*.py' -v
```

Result: **32 tests passed**, actual exit0. The candidate was extracted once and every regular file was subsequently checked byte-for-byte against the same archive before execution. The actual `train`, `jepa`, `encoders`, `utils` and both RH56 learning modules imported from that candidate checkout.

With the existing external native expert bundle, a zero-step reset comparison against the original production adapter proved exact equality of observation56, proprio13, qpos, qvel, expert_action12 and the complete physical-state hash. Original physics/capture methods and the opposite-camera render method also have identical ASTs. No rendering/capture, rollout, collection episode, optimizer update, GPU model forward or SR evaluation was executed.

The first Linux harness attempt stopped before adapter execution because that environment did not contain pytest. Its failed state/log/exit were retained. A separate version used the already installed unittest runner on the unchanged candidate; it completed the checks above without installing into the Nix environment or relabelling the first attempt as passed.

## Review and scientific limits

Data and learning modules had separate owners, froze before independent review and passed a final scope review. Added runtime files contain no deployment-specific PID/server/Nix/license path, private key, dataset or model weights. External expert/EAI paths remain explicit.

This is integration/interface validation, not six new training runs. [Original production results](../results/rh56_relocate_20261025.json) retain their real producer hashes and known limits. In particular, the earlier RGB full-model CPU/GPU numerical-equivalence gate remains failed, the uint8/GPU switch remains default-off, and exact resume continuity or legacy BC numerical parity is not newly established here.
