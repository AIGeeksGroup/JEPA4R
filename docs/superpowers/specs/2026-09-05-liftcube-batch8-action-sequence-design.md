# LiftCube Batch-8 Ordered Action Conditioning Design

## Objective

Train one 2500-step LiftCube world-model experiment that isolates action conditioning from motion-patch heuristics. The run must use physical batch size 8 with every batch drawn from distinct episodes, preserve all ten atomic actions for each skip-10 transition, and make action order observable before latent-to-action cross-attention.

## Scope

- Use the existing 90 training and 10 validation episodes.
- Keep `skip_frame=10`, `T=3`, `K=512`, and `D=64`.
- Do not configure or run MPC.
- Do not change old checkpoint architecture or loading behavior unless a new configuration explicitly enables the new action sequence encoder.
- Do not use motion matching, motion thresholds, or patch-motion weights in this experiment.

## Batch Construction

Add a deterministic cross-episode batch sampler. Each yielded batch contains at most one window from any episode. Training batches shuffle both episode order and the window order within each episode from a seed plus epoch counter. Validation batches are deterministic. Physical batch size is 8; gradient accumulation is not a substitute because SIGReg must observe eight samples in the same forward pass.

The sampler consumes `ManiSkillSequenceDataset.index`, whose entries are `(episode_id, start)`, and uses every eligible index once per epoch except an incomplete final batch when `drop_last=True`.

## Loss

The experiment uses:

\[
L = L_{\mathrm{latent}} + \lambda_{\mathrm{rank}}L_{\mathrm{rank}} + 0.09L_{\mathrm{SIGReg}}.
\]

Set `flow_weight=0`. Add an explicit `ranking_motion_weighted` option. When false, ranking compares correct and negative predictions directly against the same-index target latent and never calls `match_motion_weights`; `positions_all` is unnecessary for ranking. Existing motion-weighted configurations retain their old behavior.

The negative action sequence is the current batch roll. Because batches contain distinct episodes, every negative comes from a different episode.

## Ordered Action Encoder

The current learned positional embedding has only about two percent of the action-token RMS, and cross-attention nearly removes reversal differences. The new configuration therefore enables:

1. Per-action normalization and the existing MLP projection.
2. A fixed sinusoidal position table with order-one amplitude for positions 0 through 9.
3. One pre-norm Transformer encoder layer applied independently to each 10-action chunk.
4. A final LayerNorm, returning all ten contextual action tokens.
5. Existing latent-to-action cross-attention in every predictor block.

Default action settings keep `token_encoder_layers=0` and learned positions, preserving the state-dict structure expected by old checkpoints. The new experiment alone enables the sequence encoder and sinusoidal positions.

## Evaluation

After tests and a batch-8 CUDA preflight, train in a new run directory:

`/workspace/users/lewm/runs/4d-jepa-architecture/group1_skip10_action_sequence_batch8_2500`

Save metrics, status, logs, and checkpoints every 500 steps. Evaluate `best.pt` and `last.pt` at horizons 1, 2, and 4 with:

- correct actions;
- zero actions;
- shuffled actions from other windows;
- each 10-action chunk reversed;
- the last action in each chunk repeated ten times;
- persistence.

The primary action-control criterion remains shuffled MSE at least 10 percent worse than correct MSE. Reverse and repeat-last must also change predictions in the expected direction. A result is rejected if correct prediction remains worse than persistence or if the gap is caused only by larger prediction deltas.

## Verification

- Unit tests prove episode uniqueness per batch and deterministic validation order.
- Unit tests prove unweighted ranking does not call motion matching.
- Unit tests prove reversing a chunk changes the contextual token set seen by cross-attention.
- Existing checkpoints still load strictly.
- Focused tests, full pytest, `py_compile`, and `git diff --check` pass before training.
