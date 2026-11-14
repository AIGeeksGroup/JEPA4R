"""Stage ``train_decoder`` -- latent-space reconstruction decoder D_psi (Sec. 4.4, Eq. 2).

in : data/wm/<env>/ (OGBench Cube) and the frozen world model cfg.decoder.wm (JEPA4R, epoch 10)
out: outputs/train_decoder/<env>/<decoder>/<tag>/ -- stage-A and stage-B checkpoints and metrics.json:
       val_nn_stage_a, val_nn_stage_b   validation mean NN distance (cm) after each stage (paper 0.87 -> 0.79)
"""


def run(cfg, out_dir):
    from jepa4r.decoder.train import train

    return train(cfg, out_dir)
