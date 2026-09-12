# LiftCube Joint Cost and Contact-Aware World Model Design

## Goal

Make LiftCube CEM rank candidates with a joint, lower-is-better objective that combines goal-latent distance and predicted physical state, then train an explicit cube-robot contact predictor without overwriting prior data or checkpoints.

## Verified starting point

- The 25-D proprio layout is `qpos[0:9] + qvel[9:18] + tcp_pose[18:25]`.
- Gripper gap is `abs(qpos[7]) + abs(qpos[8])`. The existing one-step proprio head reaches `0.794 mm` RMSE and must not be redesigned on the basis of the invalid old quaternion-index metric.
- The current goal-aware cost already has height, hold, stability, approach, early-close, and optional latent components. In the proprio-enabled online and ranking-audit paths no goal latent is supplied, so the latent component is currently zero.
- The model has no contact head. The validated contact sidecar contains exact `(episode_id, frame_id)` labels.
- With `T=3` and `skip_frame=10`, every training window supervises three transitions. Across the training split these targets contain 15,166 contact and 1,292 non-contact labels; the last target alone is nearly all contact, which explains the degenerate last-step probe.

## Architecture

1. Load contact labels by exact episode/frame key and return labels for all `T+1` sampled observation frames.
2. Keep cross-episode batches, but repeat windows containing non-contact or a 0-to-1 transition four times. This changes effective exposure, not the immutable compact episodes.
3. Enable a linear contact-logit head only when `loss.contact_weight > 0`. Supervise all `T` next-step logits with globally balanced binary cross-entropy. Old configurations keep the head disabled and remain strict-load compatible.
4. Extend transition and auxiliary-rollout outputs with contact probability. Recurrently predicted proprio remains unchanged.
5. Build and encode the rendered LiftCube goal cloud for proprio-enabled CEM, then pass it through `GoalSpec.demonstrated_goal_latent`.
6. Use only non-negative penalties:

   `J = lambda_latent*J_latent + lambda_height*J_height + lambda_approach*J_approach + lambda_gripper*J_gripper + lambda_contact*J_contact + lambda_hold*J_hold + lambda_stability*J_stability + lambda_early_close*J_early_close`.

   Gripper penalty uses predicted `qpos[7:9]`. Contact penalty is active when the TCP is within the approach radius, the gripper is predicted closed, or the cube is predicted above its initial height. It penalizes `1 - p(contact)`.

7. Initialize the new model from the verified step-2500 checkpoint, allowing only the new contact-head parameters to be missing. Use a fresh optimizer and a new run directory.

## Validation gates

- Unit tests prove frame alignment, phase repeats, balanced BCE, strict checkpoint initialization, goal-latent use, corrected gripper indices, and contact-cost behavior.
- GPU preflight must produce finite loss and non-zero contact-head gradients.
- A short training smoke must reduce/keep finite contact loss and write a loadable checkpoint.
- Formal training is evaluated with per-head validation metrics and the same-state model-vs-simulator ranking audit. Finite cost is not physical success.
- Transfer only code provenance, the best checkpoint, resolved config, metrics, reports, and audit summaries to `A800`.

## Non-goals

- Do not regenerate the compact RGB/proprio dataset in this iteration.
- Do not overwrite the existing step-2500 checkpoint, probe results, or ranking-audit results.
- Do not claim the joint objective works until simulator ranking and physical diagnostics improve.
