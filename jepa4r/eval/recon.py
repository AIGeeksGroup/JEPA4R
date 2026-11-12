"""Reconstruction evaluation (stage eval_recon): Tab. 3, Tab. 4a, the open-loop figure, Fig. 1a.

per frame  decode encoder latents of cfg.decoder.eval.per_frame.num_frames validation frames; also decode the
           latent of another frame (shuffled control)
open loop  for cfg.decoder.eval.open_loop.num_episodes episodes: encode the context frames at env steps 0/5/10,
           roll the predictor forward with the episode's own actions (imagination), with another episode's
           actions (control), and decode encoder latents of the true frames (true latents); decode every step
Metrics: jepa4r/metrics/pointcloud.py, in cm in the world frame. Output keys: jepa4r/stages/eval_recon.py.
"""


def evaluate(cfg, out_dir) -> dict:
    raise NotImplementedError("TODO(integrate): per-frame and open-loop reconstruction metrics")
