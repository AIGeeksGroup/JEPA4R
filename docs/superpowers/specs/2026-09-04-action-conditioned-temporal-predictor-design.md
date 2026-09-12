# Action-Conditioned Temporal Predictor Design

## Goal

Make the LiftCube world model learn action-conditioned transitions at every observed time step, instead of using only the final action after temporal context has already been collapsed.

## Original diagnosis

Before this change, the path accepted \`actions [B,T,7]\` but predicted with \`actions[:, T-1:T]\`, returned only the last-frame temporal readout, and passed a single action token to a predictor whose temporal length was one. With the default \`fusion=add\`, the action was broadcast as a global latent bias. The existing \`MSE + SIGReg\` objective did not require correct action/state pairing to outperform shuffled actions.

## First-version design

1. Keep the public input contract \`points [B,T+1,K,3]\` and \`actions [B,T,7]\`.
2. Encode the first \`T\` point-cloud frames into \`latent1 [B,T,K,D]\` and the action sequence into \`action_emb [B,T,D]\`.
3. Add a causal temporal action-conditioned predictor that keeps the spatial token axis \`K\` and predicts \`pred_all [B,T,K,D]\`, where output index \`t\` predicts \`latent1[:,t+1]\` from the prefix ending at \`t\` and action \`a_t\`.
4. Reuse the predictor's per-layer action fusion: \`add\` broadcasts the aligned action embedding over spatial tokens, while \`modulate\` applies shift/scale conditioning. The implemented default remains \`add\`; no new AdaLN-zero gates or checkpoint parameters were added. Do not use future observation tokens when producing an earlier prediction.
5. Train on all \`T\` transitions with \`MSE(pred_all, latent1[:,1:])\` plus the existing SIGReg term. Preserve \`pred=pred_all[:,-1]\` and existing loss keys for current callers; expose \`pred_all\` for diagnostics.
6. Keep rollout input/output shapes unchanged. For each rollout step, combine the previous \`T-1\` context actions with the current action into \`step_actions [B,T,A]\`, predict, and only then shift the action history. This keeps training and inference action alignment identical.

## Compatibility and limits

- The parameter structure is unchanged, so existing checkpoints load strictly. They have not learned the new all-transition objective, so a fresh or continued training run is still required for a meaningful comparison.
- The first version does not regenerate data or add robot point flows. That is a later PointWorld-inspired experiment.
- \`fusion=add\` and \`fusion=modulate\` remain accepted. The Group 1 checkpoint evaluated below uses \`fusion=add\`.

## Verification

- Unit tests prove that all action time positions affect the corresponding prediction path, future observations cannot affect earlier predictions under the causal mask, and the public \`forward\`/\`rollout\` shapes remain unchanged.
- Existing tests must continue to pass.
- A short training/evaluation run must compare correct, zero, and shuffled actions using the existing rollout evaluator. The desired diagnostic is a measurable correct-vs-shuffled target-error gap, not merely a nonzero prediction delta.

## Empirical result

The original Group 1 checkpoint at step 8500 was continued independently to step 8800, then evaluated over all 804 validation windows with horizons 1, 2, 4, and 8. The evaluator supplies the real \`T-1\` context-action history to every correct, zero, and shuffled rollout.

| Horizon | Correct MSE | Zero MSE | Shuffled MSE | Persistence MSE | Shuffled minus correct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.265101e-6 | 6.478137e-6 | 3.343555e-6 | 3.707396e-7 | 2.40% |
| 2 | 4.303200e-6 | 8.057253e-6 | 4.353270e-6 | 3.762068e-7 | 1.16% |
| 4 | 7.732084e-6 | 1.400898e-5 | 7.784270e-6 | 3.877103e-7 | 0.67% |
| 8 | 1.891314e-5 | 2.451697e-5 | 1.902540e-5 | 4.106800e-7 | 0.59% |

Supplying real context actions improves correct-action MSE by about 1.3%-3.6% relative to the earlier zero-history evaluation. Correct actions now beat shuffled actions consistently, but only by 0.6%-2.4%, while persistence remains about 9-46 times better than the learned rollout. This is evidence of weak action-state pairing, not a strong action-conditioned dynamics result. The next experiment should change the learning signal or dynamic representation rather than repeat the causal multi-step wiring.

Artifacts:

- Checkpoint: \`/workspace/users/lewm/runs/4d-jepa-architecture/group1_action_conditioned_smoke_20260904/last.pt\`
- Metrics: \`/workspace/users/lewm/runs/4d-jepa-architecture/group1_action_conditioned_smoke_20260904/rollout_metrics_context_actions.json\`

## References

- LeWM: https://arxiv.org/abs/2603.19312 and \`/workspace/users/lewm/baselines/lewm-original/module.py\`, where action is a per-time-token AdaLN condition and training predicts multiple next embeddings.
- PointWorld: https://arxiv.org/abs/2601.03782 and \`/tmp/pointworld-src/pointworld/base.py\`, where robot point trajectories jointly interact with scene points and the target is scene point flow.
