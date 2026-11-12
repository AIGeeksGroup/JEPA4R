"""PointCloudDecoder: set generator -> frozen backbone (+ LoRA in stage B) -> head; z [B, 192] -> P_hat [B, 1200, 6]."""


def build_decoder(decoder_cfg):
    raise NotImplementedError("TODO(integrate)")


def load_decoder(env, decoder, tag="default", stage="b"):
    """Checkpoint from outputs/train_decoder/<env>/<decoder>/<tag>/."""
    raise NotImplementedError("TODO(integrate)")
