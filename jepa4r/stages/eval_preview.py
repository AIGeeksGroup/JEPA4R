"""Stage ``eval_preview`` -- decoded look-ahead of the policy in the twin (Sec. 4.3, Sec. 5.6).

The predictor rolls the policy head's action blocks forward and the decoder previews each executed 25-step segment
as point clouds (cfg.env.preview). Metrics (Sec. 5.6 reports no twin numbers; paper/expected/twin.yaml):
  preview_nn                 decoded preview vs. observed clouds (cm)
  preview_cube               cube position error of the preview (cm)
  cube_to_target_success     cube distance to its target at the end of the last
                             imagined segment, successful episodes (cm)
  cube_to_target_failure     same, failed episodes (cm)
in : the twin-trained world model, policy head (train_policy) and decoder (train_decoder) of cfg.env
out: outputs/eval_preview/<env>/<decoder>/<tag>/metrics.json, 50 twin evaluation episodes
"""


def run(cfg, out_dir):
    raise NotImplementedError("TODO(integrate): look-ahead preview in the twin (docs/reproduce.md, Open Items)")
