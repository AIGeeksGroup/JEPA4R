"""Thin Cube3 registration for the shared Native SR runner."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in Path(__file__).resolve().parents:
    if (candidate / "checkpoint_encoder.py").is_file():
        ROOT = candidate
        break
CASE = ROOT / "cube23_distance_gciql_v1"
EXT = CASE / "five_group_extension_v1"
CUBE3_CROP_BOX = ((-0.25, -0.50, -0.02), (0.75, 0.50, 0.90))

GROUPS = {
    "G08": {
        "cube_count": 3, "state_dim": 46, "current_dim": 211, "goal_dim": 201,
        "max_episode_steps": 1000, "mode": "rgb", "record_capture_receipts": False,
        "wm": ROOT / "checkpoints/2d_epoch10/weights_epoch_10.pt",
        "wm_sha256": "4e16a6c5c58e8c44728d153e4ca7149d05676973379d99774ff7db2e08bbfd4e",
        "training": EXT / "configs/G08_cube3_rgb.json",
        "training_sha256": "6d717c2123217020fec593367f3eecb42e01dfe203e9aae8d5a38733045f55b5",
        "acceptance": EXT / "acceptance/G08_full_acceptance.json",
        "acceptance_sha256": "35ff2b01f6f6efbe21c24ff136ef05464ac8eb6f5ba32b7d0f4a91b35ca2dc86",
        "gate": EXT / "gates/G08/next_batch.npz",
        "gate_sha256": "ec8f4357f3399f773c3033aa3fe8150e21f7b1f296c85f537790ebec5802a476",
        "final": EXT / "runs/G08_cube3_rgb/checkpoints/update_001000000",
        "manifest_sha256": "a0893d3b0a2d1684b23c4520a2ae55ac434bfde14432167a5c8c1800bb9d414f",
    },
    "G09": {
        "cube_count": 3, "state_dim": 46, "current_dim": 211, "goal_dim": 201,
        "max_episode_steps": 1000, "mode": "pcd", "record_capture_receipts": True,
        "pcd_crop_box": CUBE3_CROP_BOX,
        "wm": ROOT / "checkpoints/3d_epoch10/weights_epoch_10.pt",
        "wm_sha256": "07bb8d2160b1a0fecda35bba3aede4790071540a6d7a9c34b92d89ae437a04b6",
        "training": EXT / "configs/G09_cube3_pcd.json",
        "training_sha256": "16aa74533db3187e581eb3aee6ccf90d6930c37119e29320983c425125216893",
        "acceptance": EXT / "acceptance/G09_full_acceptance.json",
        "acceptance_sha256": "552b3e8e04f61233cc601651762f918c999769c88a03f25b018710a74398343c",
        "gate": EXT / "gates/G09/next_batch.npz",
        "gate_sha256": "f535347e5605de7052cd11790c716e89eb93abc4c2f0d1f8cfd929922d266e8d",
        "final": EXT / "runs/G09_cube3_pcd/checkpoints/update_001000000",
        "manifest_sha256": "fcae078a0350bde4b02a5e35cedcf47445751628805c69539478f7eeb0557f56",
    },
}

def install(core):
    core.GROUPS.update(GROUPS)
    return core

def contract():
    return {group: {key: cfg[key] for key in
            ("cube_count", "state_dim", "current_dim", "goal_dim", "max_episode_steps", "mode")}
            for group, cfg in GROUPS.items()}

