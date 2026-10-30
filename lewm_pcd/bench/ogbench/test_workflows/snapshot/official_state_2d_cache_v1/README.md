# Official Triple state-derived 2D LeWM embeddings

This producer prepares only frozen192-D FP32 features for official3000 train
and300 validation episodes. Every episode retains all1001 canonical raw rows:
1000 transition-source rows plus the actual raw1000 terminal. The data source
is the accepted official NPZ, never the migrated8-frame smoke or custom600
trajectory data. No GCIQL, world-model update, SR, or3D encode is dispatched.

```bash
"$PY" -B "$H/produce.py" --plan "$H/producer_plan.json"             # paths only; no model/render
"$PY" -B "$H/produce.py" --plan "$H/producer_plan.json" --execute --pilot
"$PY" -B "$H/produce.py" --plan "$H/producer_plan.json" --execute   # after Root pilot acceptance
```

Raw qpos35/qvel32 are decompressed once into shared read-only NPY maps. Two
spawned renderer workers restore the unchanged official static pose and call
the original renderer. The appearance follows approved collection: states,
visualize_infoFalse, default original palette/opaque materials; front_pixels
copies original front optics plus positive local-X5°, no pullback, FOV45,
RGB224, AA4, no crop. IDs are resolved by camera name. No ctrl/mocap/contact
replay is claimed. Original proprio19 remains in original observation data.
Only the necessary six pilot diagnostic RGB frames are saved, not the roughly
497GB full RGB sequence. Two bounded episode futures and B32 encode cap RAM;
nice10/eight CPU cores/Torch CPU1 preserve priority for the separate CPU SR.

The unchanged checkpoint_encoder factory/preprocessor/encode loads the accepted
Triple2D epoch10 checkpoint. Only encoder+projector receive one frame(T=1).
No AMP or precision settings are introduced. A short tail9 is padded23 to B32
and discarded afterward: each cache has1001 real rows,1024 encoder rows.

Code defaults pixel preprocessing to CPU. This plan explicitly authorizes
`auto_validated`: first use the same32 real images to compare original CPU
preprocessing with uint8-to-CUDA followed by that same original preprocessor.
Declared thresholds are pixel atol1e-6/rtol0, latent atol1e-5/rtol1e-5, all
finite, and at least5% median speed gain. Warmup3 and20 timed batches/path
include transport+preprocess+encode and GPU-tail synchronization, excluding
rendering. CUDA is selected only on both gates; otherwise the negative result
is kept and CPU selected. OOM stops the job. Immutable preprocess_selection.json
is created before either pilot episode's canonical features are written; full
production reuses its SHA, effective protocol and both pilot episode caches.

Each episode publishes latent.npy(1001,192) and receipt.json only after atomic
write/readback. Protocol binds source/model/camera/preprocessor/code/selected
device and selection SHA, while run limits/output paths are separate. Shared
geometry has its own deterministic source-array contract. NPZ files are only
checked against accepted stat records, not rehashed. Cache totals are3303300
real rows, approximately2.54GB. Per-episode files preserve raw IDs for future
current/next/goal lookup without changing any downstream input contract.

Pilot covers train0/val0 plus each split's raw1001 witness, canonical999/1000/
1001 boundaries, cubeXYZ geometry against original observation fields, and
one fixed-pose RGB invariance check after changing missing ctrl/target mocap.
Full-sequence RGB and recomputed contact observations are never substituted.

cache_seal.json is explicitly `producer_closed_pending_outer_and_independent_audit`,
ready_for_downstreamFalse. Only Root may publish READY after actual childexit0,
drained workers and an independent feature-only SHA/finite/shape/ID audit0.
Original data, source, checkpoints, old runs and current SR are unchanged.
