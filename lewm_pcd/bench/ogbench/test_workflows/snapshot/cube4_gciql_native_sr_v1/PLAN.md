# G11/G12 final1M Native SR first

The user explicitly selected G11 RGB and G12 PCD final1,000,000-update checkpoints. Host: `103032Test`. Reuse the already accepted final policies and their matching Quad LeWM epoch10 encoders. Do not retrain or modify the existing training, caches, checkpoints or results.

Execute serially: G11 four-step diagnostic → metadata audit → full250 → independent audit → G12 same sequence. Diagnostic episodes never contribute to SR. Official Quad tasks1–5 each have50 episodes, 1,000 policy-step limit, original seed970000000+task*100000+index, official success, temperature0, no Gaussian and no extra policy warmup.

Use native192 embeddings, current192+proprio19=211, goal192+four-cubeXYZ12=204. Every policy call really encodes goal then current; do not substitute an offline feature for a newly rendered scene. Preserve the original action5→control7, environment physics, reset behavior, and M5 Torch1/BLAS4 execution.

Only new N4 registration, the two existing environment dictionaries and the approved camera binding need adaptation. Camera matches the Quad encoder training: original front plus local-X+5° and local-Z pullback0.10m; FOV45/RGB224AA4. PCD uses1200 points, original geometry preprocessing and crop[[-.25,-.5,-.02],[.75,.5,.9]]. No Double crop or original front-only view.

Success means each role's real native/child/parent exits0, drained owned groups,250 unique formal episodes, finite5-D actions,211/204 input records, true2N encoder counts, matching final-policy fingerprint and strict source/config bindings, followed by an independently counted per-task and overall SR. A low SR is a valid result; no score threshold is added. Preserve the original evaluator's known reset/RNG limits; do not claim historical strict paired scenes.

PLDM512 interfaces continue CPU preparation in their separate case. PLDM encoding/training/SR can follow only after both Quad roles' actual evaluation acceptance. Keep parameter/source/CLI/log/exit records and one ETA-based follow-up; do not run another GPU task concurrently or restart a role on an old output directory.
