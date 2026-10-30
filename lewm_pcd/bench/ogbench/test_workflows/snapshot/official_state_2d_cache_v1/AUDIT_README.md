# Independent CPU cache auditor

Scope: `audit_cache.py`, `test_audit_cache.py`, and this document belong to
the auditor owner. Production producer/renderer/model code is not modified.

Implementation plan: first write one complete positive synthetic cache and
one real latent-byte corruption case; observe the missing-auditor failures;
implement only metadata/provenance/new-feature checks; run those two cases;
freeze the auditor for Root's independent review. No model, renderer, GPU,
worker, precision, or optimizer test matrix is part of this audit.

```bash
"$PY" -B audit_cache.py --cache-dir "$CACHE" --plan "$PLAN"
```

Default output is the fresh `$CACHE/outer_audit_v1` directory. It contains
`status.json`, `progress.jsonl`, `report.json` or `failure.json`, and the
auditor's actual exit-code file. It refuses an existing output directory.
Use an explicit new `--output-dir` for a later independent attempt; old
audits and feature files remain unchanged. Root must also record the real
auditor subprocess exit before producing final READY.

The CLI requires all3000 train and300 validation episodes. There is no
subset/count override in the production CLI. Each fixed path is
`split/episode_%06d/{latent.npy,receipt.json}`. Every new latent is fully
SHA-hashed and checked for strict NPY length, shape1001x192, native FP32,
finite values, stable before/after stats, and receipt identity. Canonical
raw IDs are little-endian int64; the true terminal is episode*1001+1000.
Manifest entries must cover every episode in order and bind each receipt's
actual SHA. Base and effective protocol JSON are independently reconstructed
from plan/source pins and canonical-hashed. The selected CPU/CUDA mode and
immutable preprocessing-selection receipt SHA are bound to every episode.
CUDA selection must carry the original finite/numeric/+5% decision receipt;
the auditor never repeats the NN test or changes its locked thresholds.
The unpacked-source contract is separately canonical-hashed and remains
independent of feature preprocessing selection. Camera pose/name/AA4/opaque-arm witnesses, source
array receipt SHA, checkpoint/preprocessor/producer IDs, and padding32
contract1001+23=1024 are checked.

Old NPZ files and already verified unpacked qpos/qvel source arrays are
stat-checked only. The auditor does not load/hash their contents or read
model weights. It checks code pins and small acceptance/protocol/receipt
JSON identities. Source/evidence stats are rechecked after the whole audit.
The positive test deliberately uses non-NPZ/non-NPY sentinels for old/source
files, demonstrating that the auditor never loads them.

RGB hashes and zero-error geometry witnesses are validated as frozen
producer receipts. The auditor does not render RGB or reproduce encoder
outputs; it therefore does not establish independent visual/feature
numerical equivalence. Existing producer smoke/witnesses remain separate
evidence. No online-scene encoding or SR is substituted by this cache.

The producer seal must still have ready_for_downstream=false. Successful
auditor output also leaves it false; only Root may close actual producer,
outer supervisor and independent audit gates and then generate final READY.

```powershell
python -X utf8 -B coordination/cube_downstream_101732Test_20261017_v1/official_state_2d_cache_v1/test_audit_cache.py -v
```

These are two CPU/file tests only: a positive3-episode fixture (test-only
internal count patch, each still1001x192) and a one-byte latent corruption
negative case. The production episode counts stay fixed in the audited
source. Actual red/green receipts and source SHA values are recorded in
AUDIT_FREEZE.json after the final run.

The auditor explicitly refuses Python optimization (-O/PYTHONOPTIMIZE),
which would remove assertion checks, and rejects cache-root/split/episode
symlinks that could redirect the canonical new-cache paths elsewhere.
