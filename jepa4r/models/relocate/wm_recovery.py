"""Opt-in recovery wiring for the existing Adroit SPT Manager and loader."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(args):
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(args.output / "checkpoints/relocate/config.yaml"), resolve=True)
    cfg["data"]["dataset"]["name"] = "$BOUND_TRAIN_CONTENT"
    source = args.data_root / ("train_pcd.h5" if args.model == "jepa4r" else "train_rgb.h5")
    path = ROOT / "envs/adroit/hdf_binding.py"
    spec = importlib.util.spec_from_file_location("_adroit_recovery_hdf_binding", path)
    binding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(binding)
    data = binding.hdf_dependencies(source)
    return dict(
        model=dict(
            method=args.model,
            model_config=cfg["model"],
            adapter_sha256=_sha(ROOT / "policy/bc/relocate/adroit_learning.py"),
            recovery_sha256=_sha(__file__),
            loader_sha256=_sha(ROOT / "models/relocate/recovery.py"),
            upstream_train_sha256=_sha(ROOT / "models/world_model.py"),
        ),
        data=dict(content_sha256=data["content_sha256"], audit_sha256=_sha(args.data_root / "audit.json")),
        config=hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        split="original train-file normalization then original clip-random90/10",
        action_dim=30,
    )


def validate_resume(path, identity):
    from jepa4r.utils.training_state import validate_identity

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    meta = json.loads(path.with_suffix(".ckpt.json").read_text())
    validate_identity(identity, meta["identity"])
    if (
        meta.get("schema") != "sealed-lightning-recovery-v1"
        or meta.get("actual_exit_code") != 75
        or meta.get("phase") != "paused_for_independent_resume"
        or _sha(path) != meta["sha256"]
    ):
        raise ValueError("resume needs the unchanged sealed partial training checkpoint")
    return path


@contextmanager
def training_recovery(args, observed, counts, restored_counts):
    import numpy as np
    import torch
    import stable_pretraining as spt
    from jepa4r.models.relocate.recovery import stateful_loader_class, make_lightning_recovery_callback

    if args.mode != "smoke" or args.model not in ("jepa4r", "lewm", "pldm"):
        raise ValueError("Lightning recovery is an explicit bounded JEPA4R/LeWM/PLDM interface")
    original_loader = torch.utils.data.DataLoader
    original_manager = spt.Manager

    def evidence():
        return {name: observer.history_state_dict() for name, observer in observed["optimizer_observers"].items()}

    def restore_evidence(saved):
        if set(saved) != set(observed["optimizer_observers"]):
            raise ValueError("restored optimizer observer identities differ")
        for name, state in saved.items():
            observed["optimizer_observers"][name].load_history_state_dict(state)
            restored_counts[name] = int(state["count"])
            counts[name] = int(state["count"])

    def manager(*manager_args, **manager_kwargs):
        trainer = manager_kwargs.get("trainer")
        if trainer is None:
            raise ValueError("recovery requires the original named Trainer connection")
        identity = _identity(args)
        directory = args.output / "recovery"
        callback = make_lightning_recovery_callback(
            identity,
            directory,
            pause_after_batches=args.pause_after_batches,
            evidence_provider=evidence,
            evidence_restorer=restore_evidence,
        )
        (directory / "IDENTITY.json").write_text(
            json.dumps(
                dict(
                    identity=identity,
                    startup_seed=3072,
                    manager_seed_argument=manager_kwargs.get("seed"),
                    scope="explicit bounded recovery; original model/loss/optimizer/scheduler",
                ),
                indent=2,
            )
        )
        if args.resume_checkpoint is not None:
            manager_kwargs["ckpt_path"] = str(validate_resume(args.resume_checkpoint, identity))
        manager_kwargs["weights_only"] = False
        trainer.callbacks.append(callback)
        return original_manager(*manager_args, **manager_kwargs)

    # Fix the RNG at construction (seed 3072), so that separately started processes build the
    # same model; the shared callback restores the resumed RNG after the validation sanity
    # check.
    random.seed(3072)
    np.random.seed(3072)
    torch.manual_seed(3072)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3072)
    torch.utils.data.DataLoader = stateful_loader_class(original_loader)
    spt.Manager = manager
    try:
        yield
    finally:
        torch.utils.data.DataLoader = original_loader
        spt.Manager = original_manager
