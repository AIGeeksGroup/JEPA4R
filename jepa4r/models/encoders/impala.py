"""Impala CNN, PLDM's encoder on OGBench Cube (supp. sec:supp-baselines: "on Cube, PLDM uses an Impala encoder").

The paper's Cube PLDM is the OGBench-cube branch's PLDM: the PLDM authors' source at commit 1bd7e56, vendored by that
branch (third_party/pldm), with this encoder built as its pldm/method.py builds it
(OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/pldm/method.py:52): ImpalaEncoder(input_channels=3, final_ln=True),
residual stacks (16, 32, 32) x 2 blocks, ReLU, Linear(2592, 512), LayerNorm. Input RGB [B, 3, 65, 65] (PLDM's Cube
input, method.py:7 IMAGE_SIZE 65, not the 224 x 224 of the other image encoders); forward(x).encodings is [B, 512].
"""
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3] / 'third_party' / 'pldm'


def build(enc_cfg):
    from jepa4r.baselines.cube.pldm import method

    method._official(SOURCE)   # the vendored author source on sys.path; refuses another pldm package
    from pldm.models.encoders.impala import ImpalaEncoder

    return ImpalaEncoder(input_channels=int(enc_cfg.hparams.input_channels), final_ln=bool(enc_cfg.hparams.final_ln))
