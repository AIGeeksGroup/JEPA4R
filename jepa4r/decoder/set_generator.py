"""Set generator (new in JEPA4R, Sec. 4.4): replaces the backbone's image input.

256 learnable queries -> 3 residual MLP blocks modulated by z_t through FiLM [perez2018film]
-> 16 x 16 grid of 1,024-d tokens (cfg.decoder.set_generator).
"""


def build_set_generator(decoder_cfg):
    raise NotImplementedError("TODO(integrate)")
