"""Stage ``eval_recon`` -- decode frozen and imagined latents: Tab. 3, Tab. 4a, the open-loop figure, Fig. 1a.

in : outputs/train_decoder/<env>/<decoder>/<tag>/ (stage-B checkpoint), the frozen world model, the validation
     split of data/wm/<env>/
out: outputs/eval_recon/<env>/<decoder>/<tag>/
  metrics.json (cm; outlier_pct in %)
    per frame, 4,096 validation frames:
      nn_mean nn_median nn_p90 outlier_pct cube_err
      shuffled_nn                     decode the latent of another frame instead (paper 12.31)
    open loop, 64 episodes, context frames at env steps 0 / 5 / 10, horizon 200 env steps:
      imag_nn@{15,100,200}            decoded imagination
      true_nn@{15,100,200}            decoded encoder (not predicted) latents
      other_actions_nn@{15,100,200}   imagination driven by another episode's actions
      imag_cube@{15,100,200}          cube position error of the imagination
      imag_nn_max                     max over env steps of the mean imagination error (paper: within 1.1)
  curves.npz           env_step, imag_nn, true_nn, imag_cube, true_cube (mean of 64 episodes; the data of Fig. recon)
  rollout_example.npz  one episode, observed vs decoded imagined clouds (the data of Fig. 1a)
"""


def run(cfg, out_dir):
    from jepa4r.eval.recon import evaluate

    return evaluate(cfg, out_dir)
