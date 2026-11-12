"""Decoder loss, Eq. 2:  L_dec = CD(P_hat, P) + SW_2^2(P_hat, P) + 0.1 * L_rgb(P_hat, P)

CD     symmetric Chamfer distance on xyz
SW_2^2 squared sliced Wasserstein-2 distance on xyz [bonneel2015sliced] (cfg.decoder.loss.sw_projections)
L_rgb  L1 colour error of nearest-neighbour matched points
"""


def decoder_loss(pred, target, loss_cfg):
    raise NotImplementedError("TODO(integrate)")
