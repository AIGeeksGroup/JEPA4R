"""Reload a Franka PCD smoke checkpoint and run one finite model batch."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import stable_worldmodel as swm
import torch

from utils import PointCloudPreprocessor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_filename(epoch: int) -> str:
    if epoch < 1:
        raise ValueError(f"epoch must be positive, got {epoch}")
    return f"weights_epoch_{epoch}.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_dir = swm.data.utils.get_cache_dir(sub_folder="checkpoints")
    filename = checkpoint_filename(args.epoch)
    checkpoint_path = checkpoint_dir / args.checkpoint / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    dataset = swm.data.HDF5Dataset(
        path=args.dataset,
        frameskip=5,
        num_steps=4,
        keys_to_load=["pcd_xyz", "pcd_rgb", "action", "observation"],
    )
    sample = PointCloudPreprocessor()(dataset[0])
    batch = {
        key: torch.nan_to_num(value.unsqueeze(0), 0.0)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }

    model = swm.wm.utils.load_pretrained(f"{args.checkpoint}/{filename}").eval()
    with torch.no_grad():
        output = model.encode(batch)
        prediction = model.predict(
            output["emb"][:, :3],
            output["act_emb"][:, :3],
        )
        target = output["emb"][:, 1:]
        pred_loss = (prediction - target).pow(2).mean()

    tensors = {
        "embedding": output["emb"],
        "action_embedding": output["act_emb"],
        "prediction": prediction,
        "pred_loss": pred_loss,
    }
    for name, value in tensors.items():
        if not torch.isfinite(value).all():
            raise AssertionError(f"{name} contains NaN/Inf")

    lines = [
        f"checkpoint={checkpoint_path}",
        f"sha256={sha256_file(checkpoint_path)}",
        f"pcd_shape={tuple(batch['pcd'].shape)}",
        f"action_shape={tuple(batch['action'].shape)}",
        f"embedding_shape={tuple(output['emb'].shape)}",
        f"prediction_shape={tuple(prediction.shape)}",
        f"pred_loss={float(pred_loss):.9g}",
        "status=PASS",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
