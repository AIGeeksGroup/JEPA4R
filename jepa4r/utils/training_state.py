"""The complete training state for resuming a run on fixed recorded batches (used by the resume checks); no
training math here."""

import copy
import hashlib
import json
import os
from pathlib import Path
import random


def validate_identity(expected, actual):
    required = {"model", "data", "config", "split", "action_dim"}
    if not required <= set(expected) or expected != actual:
        raise ValueError("training state model/data/config/split/action identity differs")


class FixedBatchCursor:
    """An already-materialized real batch stream, including its next batch ID."""

    def __init__(self, batch_ids):
        self.batch_ids = list(batch_ids)
        if not self.batch_ids or len(set(self.batch_ids)) != len(self.batch_ids):
            raise ValueError("unique explicit fixed batch IDs are required")
        self.position = 0

    def take(self):
        if self.position >= len(self.batch_ids):
            raise StopIteration
        value = self.batch_ids[self.position]
        self.position += 1
        return value

    def state_dict(self):
        return dict(
            batch_ids=self.batch_ids.copy(),
            position=self.position,
            next_batch_id=self.batch_ids[self.position] if self.position < len(self.batch_ids) else None,
        )

    def load_state_dict(self, state):
        pos = state.get("position")
        if (
            state.get("batch_ids") != self.batch_ids
            or type(pos) is not int
            or not 0 <= pos <= len(self.batch_ids)
            or state.get("next_batch_id") != (self.batch_ids[pos] if pos < len(self.batch_ids) else None)
        ):
            raise ValueError("fixed batch stream/cursor/next ID differs")
        self.position = pos


def capture_rng():
    import numpy as np
    import torch

    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("saved CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_torch_state(path, *, model, optimizers, schedulers, scaler, identity, cursor, progress):
    import torch

    validate_identity(identity, identity)
    payload = dict(
        schema="cube_fixed_batch_training_state_v1",
        identity=copy.deepcopy(identity),
        model=model.state_dict(),
        optimizers={k: v.state_dict() for k, v in optimizers.items()},
        schedulers={k: v.state_dict() for k, v in schedulers.items()},
        scaler=None if scaler is None else scaler.state_dict(),
        rng=capture_rng(),
        cursor=cursor.state_dict(),
        progress=copy.deepcopy(progress),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A manifest is published last. Partial writes and failed attempts remain unaccepted.
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = path.with_suffix(path.suffix + ".json")
    with manifest.open("x", encoding="utf8") as stream:
        json.dump(
            dict(
                schema=payload["schema"],
                sha256=_sha(path),
                identity=identity,
                progress=progress,
                cursor=payload["cursor"],
                actual_exit_code=0,
            ),
            stream,
            indent=2,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def restore_torch_state(path, *, model, optimizers, schedulers, scaler, identity, cursor):
    import torch

    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf8"))
    validate_identity(identity, manifest["identity"])
    if manifest.get("actual_exit_code") != 0 or _sha(path) != manifest["sha256"]:
        raise ValueError("training checkpoint checksum/exit differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_identity(identity, payload["identity"])
    if payload.get("schema") != "cube_fixed_batch_training_state_v1":
        raise ValueError("training checkpoint schema differs")
    if set(payload["optimizers"]) != set(optimizers) or set(payload["schedulers"]) != set(schedulers):
        raise ValueError("optimizer/scheduler identities differ")
    actual = model.state_dict()
    if set(actual) != set(payload["model"]) or any(actual[k].shape != v.shape for k, v in payload["model"].items()):
        raise ValueError("model state keys/shapes differ")
    checked_cursor = FixedBatchCursor(cursor.batch_ids)
    checked_cursor.load_state_dict(payload["cursor"])
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("AMP scaler identity differs")
    # Validate all bindings before mutating model or optimizer state.
    model.load_state_dict(payload["model"], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(payload["schedulers"][name])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    cursor.load_state_dict(payload["cursor"])
    restore_rng(payload["rng"])
    return payload["progress"]


def lightning_state_callback(identity, cursor=None):
    """Add identity/RNG/fixed-batch cursor to Lightning's existing full checkpoint."""
    import lightning as pl

    validate_identity(identity, identity)

    class SmokeState(pl.pytorch.callbacks.Callback):
        def __init__(self):
            self.pending = None

        @property
        def state_key(self):
            return "CubeFixedBatchSmokeState"

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint["cube_smoke_state"] = dict(
                identity=copy.deepcopy(identity),
                rng=capture_rng(),
                cursor=None if cursor is None else cursor.state_dict(),
            )

        def on_load_checkpoint(self, trainer, pl_module, checkpoint):
            value = checkpoint["cube_smoke_state"]
            validate_identity(identity, value["identity"])
            self.pending = value

        def on_train_start(self, trainer, pl_module):
            if self.pending is not None:
                if cursor is not None:
                    cursor.load_state_dict(self.pending["cursor"])
                restore_rng(self.pending["rng"])
                self.pending = None

    return SmokeState()
