"""PLDM [sobal2025pldm]: trained on our image data for 10 epochs with its own recipe; ViT-tiny/14,
Impala on OGBench Cube.

MetaWorld, Scene, Puzzle: stable-worldmodel's own PLDM training (third_party/stable-worldmodel/scripts/train/pldm.py,
config pldm.yaml, at 6f1e499 with the logger / checkpoint-interval patch), started as in the paper runs: the recipe's
config overrides
(cfg.method.train.recipe) + the env's image dataset + trainer.max_epochs. The checkpoint (config.json +
weights_epoch_<n>.pt, encoder + projector + predictor) lands in outputs/train_wm/<env>/pldm/<tag>/checkpoints/.
OGBench Cube (Impala encoder, the Cube workflow's PLDM): jepa4r/baselines/cube/pldm.
The policies read encoder + projector, 192-d (jepa4r/policy/bc/jepa_baselines.py load_pldm: CortexBench, DQC).
Interpreter: JEPA4R_PY_TRAIN (stable-worldmodel, stable-pretraining 0.1.8, Lightning).
"""

import os
import subprocess
import sys
from pathlib import Path

SWM = Path(__file__).resolve().parents[2] / "third_party" / "stable-worldmodel"


def overrides(cfg):
    """The Hydra overrides of scripts/train/pldm.py for this env, as in the paper's runs."""
    from jepa4r.config import stage_dir

    data = stage_dir("collect_wm_data", cfg.env.name) / cfg.env.wm_data.files.image
    return [
        *map(str, cfg.method.train.recipe.overrides),
        f"data.dataset.name={data}",
        "subdir=",
        "output_model_name=",
        f"trainer.max_epochs={int(cfg.method.train.epochs)}",
    ]


def train(cfg, out_dir) -> dict:
    from jepa4r.models.world_model import training_summary

    out_dir = Path(out_dir)
    # STABLEWM_HOME = the run directory: pldm.py writes its config.yaml and checkpoints to
    # $STABLEWM_HOME/checkpoints/<name>
    env = dict(
        os.environ,
        STABLEWM_HOME=str(out_dir),
        HYDRA_FULL_ERROR="1",
        PYTHONPATH=os.pathsep.join([str(SWM), os.environ.get("PYTHONPATH", "")]),
    )
    with open(out_dir / "train.log", "a") as log:
        subprocess.run(
            [
                os.environ.get("JEPA4R_PY_TRAIN", sys.executable),
                str(SWM / "scripts" / "train" / "pldm.py"),
                *overrides(cfg),
            ],
            cwd=out_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return training_summary(out_dir)
