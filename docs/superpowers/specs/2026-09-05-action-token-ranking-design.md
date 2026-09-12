# Ordered Action Tokens and Motion Ranking Design

## Goal

Increase action identifiability in the LiftCube skip-frame=10 world model without replacing its existing latent prediction objective or overwriting prior experiments.

## Current failure mode

The current two-layer action MLP initializes both Linear layers with Xavier gain 0.01. LayerNorm partially repairs the first layer's scale, but the unnormalized second layer collapses differences between actions. A GRU then compresses ten already-similar embeddings to one vector, and an output multiplier amplifies both shared bias and action differences equally. Motion weighting focuses learning spatially but does not require the predictor to prefer the correct action.

## Chosen architecture

The legacy `add` and `modulate` paths remain load-compatible. A new `cross_attention` fusion path makes four scoped changes:

1. The final Linear in each action MLP uses Xavier gain 1.0; the first Linear and its following LayerNorm are unchanged.
2. Each skip-frame chunk is represented as `S=10` action tokens with a learned within-chunk positional embedding, producing `[B,T,S,D]`. The new path does not use the GRU final hidden.
3. Every predictor block applies residual multi-head cross-attention from latent patch queries to the aligned action-token chunk before causal temporal attention.
4. Training constructs an in-batch wrong-action sequence and adds a motion-weighted relative ranking loss requiring its matched-target error to exceed correct-action error by 10%.

The principal loss remains:

\[
L = L_{\mathrm{latent+flow}} + \lambda_{\mathrm{rank}}L_{\mathrm{rank}} + 0.09L_{\mathrm{SIGReg}}.
\]

`L_rank` uses the same matched patch targets and motion weights as the flow-weighted latent loss. The target is detached only inside the ranking term so the target encoder cannot satisfy the contrastive constraint by moving both targets. The ordinary latent and flow losses retain their existing bidirectional-gradient behavior.

## Negative construction

For normal batches, wrong actions are a cyclic roll across the batch, preserving the empirical action distribution and all ten ordered actions. A batch of size one rolls transition chunks instead. This is deterministic, cheap, and sufficient for the requested short ablation; no queue, inverse-dynamics head, or uncertainty branch is added.

## Compatibility and failure handling

- Old configurations keep using `encode_chunks` and the GRU, so old checkpoints remain usable.
- The new positional parameter exists only when ordered tokens are enabled.
- `ranking_weight=0` preserves current loss behavior and does not require negative actions.
- The new run uses its own directory and refuses to overwrite it.
- Checkpoints and status are written every 500 steps using the established runner pattern.

## Experiment and acceptance criteria

Train one `group1_skip10_action_tokens_rank_2500` run for 2,500 steps on the existing 90 training trajectories and evaluate on the existing 10 validation trajectories at horizons 1, 2, and 4 with correct, zero, and shuffled actions.

Primary evidence is shuffled MSE relative to correct MSE. A meaningful success requires at least 10% degradation under shuffled actions, preferably beyond H1. The result is rejected as artificial if correct MSE deteriorates materially, persistence remains substantially better without a compensating action gap, or prediction delta grows without improved correct prediction. Motion-only subsets are reported as secondary diagnostics, not substituted for the global criterion.

## Deliberately excluded

No uncertainty head, full PointWorld scene-flow system, robot-point trajectory representation, replay queue, MPC environment, or MPC evaluation is included in this version.
