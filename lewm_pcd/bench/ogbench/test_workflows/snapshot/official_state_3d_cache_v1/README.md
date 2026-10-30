# Official Triple state poses -> frozen final3D LeWM embeddings

Only this new directory is changed. Original 2D cache, encoder factory, model, data,
PCD core and training code stay unchanged.

The accepted final 3D epoch10 checkpoint is strict-loaded by the existing
`checkpoint_encoder.py` factory. Its original CPU affine maps world XYZ by
`(xyz-(.425,0,.2))*4`, converts RGB uint8 to FP32/255, then sends a
`B32 x 1200 x 6` tensor to the frozen encoder+projector192. XYZ remains FP32.
No new GPU preprocessing formula, AMP, precision change, dynamics or training.

The renderer reuses the frozen 2D static pose/camera initializer and the original
`NativePCDFrame` with the original `PointCloudWrapper` plus the approved crop
`[[-.25,-.5,-.02],[.75,.5,.9]]`. RGB is AA4 and depth/segmentation are NoAA from
the same front+localX5 pose. 1200 points, foreground selection, voxel .005,
cube fraction .25 and original choice/permutation mathematics remain unchanged.

Official NPZ contains no PCD sampling seed. This new derivative explicitly uses
the uint32 little-endian first4bytes of SHA256 of compact JSON
`["triple_official_pcd_epoch10_v1", split, episode]`. Each episode advances its
private original current stream continuously from t0 through t1000. This is
independent of worker scheduling/resume and does not reproduce a historical
PCD stream. Sparse boundary witnesses advance the stream from t0 too.

Data are the same official 3000 train/300 val episodes and accepted read-only
qpos35/qvel32 mmap arrays produced by the 2D cache. Its original extraction
receipt/producer identity remains true and is adopted, not rewritten. Each EP
saves FP32 `(1001,192)` features, including real terminal t1000; source raw ID
t999 next is t1000, next EP starts t1001. Full RGB/PCD arrays are not persisted.
Only small gate/boundary diagnostics are saved.

This is static pose rendering, not action, contact or dynamics replay. Missing
ctrl/mocap/target state is checked with one same-pose/same-private-RNG RGB+PCD
witness. Original NPZ observations are still authoritative for proprio/contact.
Offline cache cannot replace true online current/goal scene encoding in SR.

Frozen exact CLI, in order after Root allocates the idle GPU:

1. `python -B gate.py --plan producer_plan.json --stage reference --output gate_v1`
2. Same independent process with `--stage fresh` (32 real inputs include 999/1000/1001).
3. `python -B produce.py --plan producer_plan.json --execute --pilot` (train0+val0).
4. Pilot child real0+workers drained and boundary readback accepted, then same CLI
   without `--pilot`; it adopts the two committed episodes.
5. Independent CPU all-new-feature audit plus real full child/outer exits and
   drained workers are required before READY. Producer seal remains pending.

Root owns dispatch, resource guard and READY. This functional owner does not
start a concurrent second producer or replace an active job. Two spawn render
workers, bounded2 futures, Torch1/B32/FP32, own-RSS cap12GiB and allocator .35
reuse the accepted 2D scheduling approach. Full cache has 3,303,300 canonical
rows; B32 tail9 padding is discarded and cannot count as extra data.
