"""Read-only observers around the learners (world models, BC); no other loss or control policy."""

import argparse
import copy
from contextlib import nullcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
WIDTHS = {"lewm": 606, "jepa4r": 606, "pldm": 606, "lejepa": 1566, "visreg": 2334, "vjepa": 2334}


def validate_optimizer_evidence(evidence):
    count = evidence["hook_count"]
    strict = evidence.get("require_nonzero_gradient", True)
    if not strict and not (
        evidence.get("role") == "pldm_aux_idm_zero_weight" and evidence.get("auxiliary_loss_weight") == 0.0
    ):
        raise ValueError("only the verified zero-weight PLDM auxiliary IDM may allow zero gradients")
    if count < 1 or evidence["gradient_steps"] != count:
        raise ValueError("missing nonempty finite gradient updates")
    nonzero = evidence.get("nonzero_gradient_steps", evidence["gradient_steps"])
    zero = evidence.get("zero_gradient_steps", 0)
    if nonzero + zero != count or (strict and nonzero != count):
        raise ValueError("missing nonzero representation gradients")
    if evidence["positive_lr_steps"] < 1:
        raise ValueError("no executed update had a positive learning rate")
    if strict and evidence["before_sha256"] == evidence["after_sha256"]:
        raise ValueError("trainable parameters did not change")
    steps = evidence["adam_state_steps"]
    if not steps or any(step != count for step in steps):
        raise ValueError("Adam state steps disagree with actual step hooks")
    return evidence


def optimizer_role(module, optimizer, algorithm, *, idm_enabled=None, idm_weight=None):
    """Identify ownership from actual parameter objects, never optimizer order."""
    owned = {id(p) for group in optimizer.param_groups for p in group["params"] if p.requires_grad}
    names = {id(p): "model." + name for name, p in module.model.named_parameters() if p.requires_grad}
    if owned and owned <= set(names):
        return dict(
            role="representation", require_nonzero_gradient=True, parameter_names=sorted(names[key] for key in owned)
        )
    if algorithm == "pldm" and hasattr(module, "idm"):
        names = {id(p): "idm." + name for name, p in module.idm.named_parameters() if p.requires_grad}
        if owned and owned <= set(names) and idm_enabled is True and idm_weight == 0.0:
            return dict(
                role="pldm_aux_idm_zero_weight",
                require_nonzero_gradient=False,
                auxiliary_loss_weight=0.0,
                parameter_names=sorted(names[key] for key in owned),
            )
    raise ValueError("optimizer ownership or zero-weight PLDM auxiliary contract was not verified")


def parameter_digest(parameters):
    import torch

    digest = hashlib.sha256()
    for value in parameters:
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class OptimizerObserver:
    """Read-only optimizer hooks: gradients, pre-step LR, state steps, parameter change."""

    def __init__(
        self,
        optimizer,
        *,
        role="representation",
        require_nonzero_gradient=True,
        auxiliary_loss_weight=None,
        parameter_names=None,
    ):
        self.optimizer = optimizer
        self.role = role
        self.require_nonzero_gradient = require_nonzero_gradient
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.parameter_names = parameter_names or []
        if not require_nonzero_gradient and not (role == "pldm_aux_idm_zero_weight" and auxiliary_loss_weight == 0.0):
            raise ValueError("zero gradients require the verified PLDM auxiliary role")
        self.parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        self.before = parameter_digest(self.parameters)
        self.count = 0
        self.gradient_steps = 0
        self.nonzero_gradient_steps = 0
        self.zero_gradient_steps = 0
        self.positive_lr_steps = 0
        self.learning_rates = []
        self.handles = [
            optimizer.register_step_pre_hook(self.before_step),
            optimizer.register_step_post_hook(self.after_step),
        ]

    def before_step(self, optimizer, argv, kwargs):
        import torch

        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise ValueError("optimizer requires nonempty finite gradients")
        nonzero = any(bool(torch.count_nonzero(g)) for g in grads)
        if self.require_nonzero_gradient and not nonzero:
            raise ValueError("representation optimizer requires nonzero gradients")
        self.gradient_steps += 1
        if nonzero:
            self.nonzero_gradient_steps += 1
        else:
            self.zero_gradient_steps += 1
        rates = [float(group["lr"]) for group in optimizer.param_groups]
        self.learning_rates.append(rates)
        if any(rate > 0 for rate in rates):
            self.positive_lr_steps += 1

    def after_step(self, optimizer, argv, kwargs):
        self.count += 1

    def report(self):
        steps = sorted({float(state["step"]) for state in self.optimizer.state.values() if "step" in state})
        return validate_optimizer_evidence(
            dict(
                before_sha256=self.before,
                after_sha256=parameter_digest(self.parameters),
                hook_count=self.count,
                gradient_steps=self.gradient_steps,
                positive_lr_steps=self.positive_lr_steps,
                nonzero_gradient_steps=self.nonzero_gradient_steps,
                zero_gradient_steps=self.zero_gradient_steps,
                role=self.role,
                require_nonzero_gradient=self.require_nonzero_gradient,
                auxiliary_loss_weight=self.auxiliary_loss_weight,
                parameter_names=self.parameter_names,
                adam_state_steps=steps,
                pre_step_learning_rates=self.learning_rates,
            )
        )

    def remove(self):
        for handle in self.handles:
            handle.remove()

    def history_state_dict(self):
        names = (
            "before",
            "count",
            "gradient_steps",
            "nonzero_gradient_steps",
            "zero_gradient_steps",
            "positive_lr_steps",
            "learning_rates",
            "role",
            "require_nonzero_gradient",
            "auxiliary_loss_weight",
            "parameter_names",
        )
        return copy.deepcopy({name: getattr(self, name) for name in names})

    def load_history_state_dict(self, state):
        for name in ("role", "require_nonzero_gradient", "auxiliary_loss_weight", "parameter_names"):
            if state[name] != getattr(self, name):
                raise ValueError("optimizer observer ownership differs on resume")
        count = state["count"]
        if (
            type(count) is not int
            or count < 0
            or state["gradient_steps"] != count
            or state["nonzero_gradient_steps"] + state["zero_gradient_steps"] != count
            or len(state["learning_rates"]) != count
        ):
            raise ValueError("restored optimizer observer counters differ")
        for name, value in state.items():
            setattr(self, name, copy.deepcopy(value))


def observed_trainer_type(original_trainer, observer_factory):
    """Preserve Trainer's type contract used by the unchanged SPT Manager."""

    class ObservedTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            callbacks = list(kwargs.pop("callbacks", []))
            config = next(callback.cfg for callback in callbacks if hasattr(callback, "cfg"))
            callbacks.append(observer_factory(config))
            super().__init__(*args, callbacks=callbacks, **kwargs)

    return ObservedTrainer


def validate_metrics(metrics, epochs):
    losses = [r["epoch_loss"] for r in metrics if "epoch_loss" in r]
    evals = [r for r in metrics if "eval/epoch" in r]
    expected = [i for i in range(epochs) if (i > 0 and i % 5 == 0) or i == epochs - 1]
    if len(losses) != epochs:
        raise ValueError("BC loss count differs from actual epoch budget")
    if not all(math.isfinite(float(v)) for v in losses):
        raise ValueError("BC losses must be finite")
    if [int(r["eval/epoch"]) for r in evals] != expected:
        raise ValueError("BC evaluation rounds differ from native schedule")
    if not all(math.isfinite(float(r["eval/success"])) for r in evals):
        raise ValueError("BC evaluations must be finite")
    return {
        "epochs": epochs,
        "evaluation_rounds": len(evals),
        "eval_epoch_indices": expected,
        "loss_records": len(losses),
    }


def configure_runtime():
    extra = os.environ.get("ADROIT_WM_EXTRA_SITE")
    if extra:
        sys.path.append(extra)
    paths = [ROOT.parent, ROOT / "policy/bc"]
    eai = os.environ.get("CORTEXBENCH_EAI_VC_ROOT")
    if eai:
        paths.extend(
            [Path(eai) / "cortexbench", Path(eai) / "vc_models", Path(eai) / "cortexbench/mujoco_vc/visual_imitation"]
        )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def reload_preprocessor(model, cfg, instantiate, pldm_native=None):
    if model != "pldm":
        return instantiate(cfg.data.preprocessor)
    if pldm_native is None:
        from jepa4r.baselines.relocate import train_pldm

        framework = Path(
            os.environ.get("ADROIT_FRAMEWORK_ROOT", ROOT.parent / "third_party/stable-worldmodel")
        ).resolve()
        pldm_native, _ = train_pldm.load_native(framework / "scripts/train/pldm.py")
    return pldm_native.get_img_preprocessor("pixels", "pixels", cfg.img_size)


def verify_frozen_checkpoint(model, path, pins=None):
    if model not in ("visreg", "vjepa"):
        return None
    if pins is None:
        pins = json.loads((ROOT.parent / "configs/policy/bc/relocate/DEPENDENCIES.json").read_text(encoding="utf-8"))[
            "frozen_encoders"
        ]
    expected = pins[model]
    path = Path(path)
    if path.stat().st_size != expected["bytes"]:
        raise ValueError("frozen encoder checkpoint byte count differs")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"model": model, "bytes": expected["bytes"], "sha256": digest.hexdigest()}


def pldm_training_plan(args):
    native = importlib.import_module("jepa4r.baselines.relocate.train_pldm")
    argv = [
        "--task",
        "adroit_relocate",
        "--framework-root",
        os.environ["ADROIT_FRAMEWORK_ROOT"],
        "--dataset",
        str(args.data_root / "train_rgb.h5"),
        "--output",
        str(args.output),
        "--name",
        "relocate",
        "--mode",
        "smoke" if args.mode == "smoke" else "formal",
        "--workers",
        "0" if args.mode == "smoke" else "16",
    ]
    parser = native.make_parser()
    plan = native.build_plan(parser.parse_args(argv), parser)
    return plan


def train(args):
    import torch
    import lightning as pl
    import stable_pretraining as spt
    from omegaconf import OmegaConf
    from jepa4r.data.relocate.adroit_data import receipt

    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("WM training requires CUDA")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    smoke = args.mode == "smoke"
    counts = {}
    observed = {"optimizer_observers": {}}
    handles = []
    restored_counts = {}
    clip_records = []
    original_trainer = pl.Trainer

    class Observer(pl.Callback):
        def __init__(self, model_config):
            self.model_config = model_config

        def on_fit_start(self, trainer, module):
            observed["model"] = module.model
            observed["seed_scopes"] = dict(
                configured_seed=3072,
                explicit_manager_seed=3072,
                actual_torch_initial_seed_at_fit_start=int(torch.initial_seed()),
            )
            if torch.initial_seed() != 3072:
                raise ValueError("guide WM training seed was not applied before fit")
            idm_enabled = idm_weight = None
            if args.model == "pldm":
                saved = OmegaConf.load(args.output / "checkpoints/relocate/config.yaml")
                idm_enabled = bool(saved.loss.idm.enabled)
                idm_weight = float(saved.loss.idm.weight)
                observed["auxiliary_idm_contract"] = {"enabled": idm_enabled, "weight": idm_weight}
            for index, opt in enumerate(trainer.optimizers):
                counts[str(index)] = 0

                def count(opt, argv, kwargs, index=index):
                    counts[str(index)] += 1

                handles.append(opt.register_step_post_hook(count))
                role = optimizer_role(module, opt, args.model, idm_enabled=idm_enabled, idm_weight=idm_weight)
                observer = OptimizerObserver(opt, **role)
                observed["optimizer_observers"][str(index)] = observer
                handles.append(observer)

        def on_train_start(self, trainer, module):
            observed["expected_updates"] = int(trainer.num_training_batches) * int(trainer.max_epochs)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if isinstance(outputs, dict) and "loss" in outputs and not torch.isfinite(outputs["loss"]).all():
                raise ValueError("nonfinite WM loss")

        def on_train_end(self, trainer, module):
            if smoke:
                from stable_worldmodel.wm.utils import save_pretrained

                save_pretrained(
                    module.model, run_name="relocate", config=self.model_config, filename="weights_epoch_1.pt"
                )

    pl.Trainer = observed_trainer_type(original_trainer, Observer)
    os.environ.update(STABLEWM_HOME=str(args.output), SPT_CACHE_DIR=str(args.output / "_cache"))
    try:
        if getattr(args, "recovery", False) and args.model in ("jepa4r", "lewm", "pldm"):
            from jepa4r.models.relocate.wm_recovery import training_recovery

            context = training_recovery(args, observed, counts, restored_counts)
        else:
            context = nullcontext()
        if getattr(args, "clip_evidence", False):
            from jepa4r.policy.bc.relocate.clip_evidence import observe_gradient_clipping

            clip_context = observe_gradient_clipping(clip_records, model_provider=lambda: observed.get("model"))
        else:
            clip_context = nullcontext()
        with context, clip_context:
            if args.model in ("jepa4r", "lewm"):
                import hydra

                native = importlib.import_module("jepa4r.models.world_model")
                overrides = [
                    "data=" + ("adroit_relocate_pcd" if args.model == "jepa4r" else "adroit_relocate"),
                    "model=" + ("lewm_pcd_dexart" if args.model == "jepa4r" else "lewm"),
                    "subdir=relocate",
                    "output_model_name=relocate",
                    "wandb.enabled=false",
                    "num_workers=" + str(0 if smoke else 16),
                    "loader.persistent_workers=" + str(not smoke).lower(),
                    "loader.prefetch_factor=" + ("null" if smoke else "2"),
                ]
                if smoke:
                    overrides += [
                        "loader.batch_size=2",
                        "trainer.max_epochs=1",
                        "+trainer.limit_train_batches=2",
                        "+trainer.limit_val_batches=1",
                    ]
                os.environ["RELOCATE_" + ("PCD" if args.model == "jepa4r" else "RGB") + "_TRAIN"] = str(
                    args.data_root / ("train_" + ("pcd" if args.model == "jepa4r" else "rgb") + ".h5")
                )
                train_configs = str(ROOT.parent / "configs/method/relocate/train")
                with hydra.initialize_config_dir(version_base=None, config_dir=train_configs):
                    cfg = hydra.compose(config_name="lewm", overrides=overrides)
                spt.set(
                    cache_dir=str(args.output / "_cache"),
                    cleanup={"checkpoints": False, "logs": False, "callback_artifacts": False},
                )
                native.run_random_split(cfg, None, manager_seed=int(cfg.seed), original_layout=True)
            elif args.model == "pldm":
                native = importlib.import_module("jepa4r.baselines.relocate.train_pldm")
                native.execute(pldm_training_plan(args))
            else:
                native = importlib.import_module("jepa4r.baselines.relocate.train_lejepa")
                learner = importlib.import_module("jepa4r.baselines.relocate.train_lejepa_relocate")
                original_encoder = learner.Encoder
                original_adamw = torch.optim.AdamW
                original_loader = learner.DataLoader
                effective_loader = original_loader
                gate = None
                if getattr(args, "recovery", False):
                    from jepa4r.models.relocate.recovery import stateful_loader_class
                    from jepa4r.baselines.relocate.lejepa_recovery import LejepaRecovery

                    effective_loader = stateful_loader_class(original_loader)

                    def saved_observers():
                        return {name: o.history_state_dict() for name, o in observed["optimizer_observers"].items()}

                    def restore_observers(saved):
                        if set(saved) != set(observed["optimizer_observers"]):
                            raise ValueError("LeJEPA optimizer observer identity differs")
                        for name, value in saved.items():
                            observed["optimizer_observers"][name].load_history_state_dict(value)
                            restored_counts[name] = int(value["count"])
                            counts[name] = int(value["count"])

                    gate = LejepaRecovery(
                        args.output / "recovery",
                        resume_from=args.resume_checkpoint,
                        pause_after_updates=args.pause_after_batches,
                        evidence_provider=saved_observers,
                        evidence_restorer=restore_observers,
                    )

                def loader(*a, **kw):
                    result = effective_loader(*a, **kw)
                    observed["expected_updates"] = len(result) * 10
                    return result

                def encoder(*a, **kw):
                    result = original_encoder(*a, **kw)
                    observed["model"] = result
                    observed["seed_scopes"] = dict(
                        configured_seed=3072,
                        actual_torch_initial_seed_after_model_construction=int(torch.initial_seed()),
                    )
                    if torch.initial_seed() != 3072:
                        raise ValueError("guide LeJEPA construction seed was not applied")
                    return result

                class AdamW(original_adamw):
                    def __init__(self, *a, **kw):
                        super().__init__(*a, **kw)
                        counts["0"] = 0

                        def count(opt, argv, kwargs):
                            counts["0"] += 1

                        handles.append(self.register_step_post_hook(count))
                        observer = OptimizerObserver(self)
                        observed["optimizer_observers"]["0"] = observer
                        handles.append(observer)

                learner.Encoder = encoder
                learner.DataLoader = loader
                torch.optim.AdamW = AdamW
                try:
                    parser = native.make_parser()
                    parsed = parser.parse_args(
                        [
                            "--task",
                            "adroit_relocate",
                            "--dataset",
                            str(args.data_root / "train_rgb.h5"),
                            "--output",
                            str(args.output),
                            "--name",
                            "relocate",
                            "--mode",
                            "smoke" if smoke else "formal",
                            "--workers",
                            "0" if smoke else "16",
                        ]
                    )
                    plan = native.build_plan(parsed, parser)
                    if smoke:
                        plan["native_argv"][plan["native_argv"].index("--bs") + 1] = "128"
                    if gate is None:
                        native.execute(plan)
                    else:
                        native.execute(plan, gate=gate)
                    if smoke:
                        net = observed["model"]
                        dest = args.output / "checkpoints/relocate"
                        torch.save(
                            {name: getattr(net, name).state_dict() for name in ("backbone", "head", "proj")},
                            dest / "weights_epoch_1.pt",
                        )
                finally:
                    learner.Encoder = original_encoder
                    learner.DataLoader = original_loader
                    torch.optim.AdamW = original_adamw
        if not counts or min(counts.values()) < 2:
            raise ValueError("fewer than two actual updates for a training optimizer")
        if smoke and any(v != 2 for v in counts.values()):
            raise ValueError("smoke optimizer count differs from two")
        if not smoke and any(v != observed.get("expected_updates") for v in counts.values()):
            raise ValueError("formal optimizer count differs from native loader length times ten epochs")
        optimization = {name: observer.report() for name, observer in observed["optimizer_observers"].items()}
        if set(optimization) != set(counts) or any(
            optimization[name]["hook_count"] != count for name, count in counts.items()
        ):
            raise ValueError("optimizer observers disagree")
        representation = {
            name: counts[name] for name, facts in optimization.items() if facts["require_nonzero_gradient"]
        }
        auxiliary = {
            name: counts[name] for name, facts in optimization.items() if not facts["require_nonzero_gradient"]
        }
        if not representation:
            raise ValueError("no representation optimizer has actual learning evidence")
        epoch = 1 if smoke else 10
        ckpt = args.output / "checkpoints/relocate"
        state = torch.load(ckpt / f"weights_epoch_{epoch}.pt", map_location="cpu", weights_only=False)

        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            return not torch.is_tensor(value) or bool(torch.isfinite(value).all())

        if not finite(state):
            raise ValueError("nonfinite saved state")
        receipt(
            args.output / "train.json",
            passed=True,
            actual_exit_code=0,
            optimizer_updates=min(representation.values()),
            per_optimizer_updates=counts,
            model=args.model,
            mode=args.mode,
            smoke_export_partial_epoch=smoke,
            restored_optimizer_updates=restored_counts,
            process_optimizer_updates={k: v - restored_counts.get(k, 0) for k, v in counts.items()},
            seed_scopes=observed.get("seed_scopes"),
            gradient_clip_events_process=clip_records,
            representation_optimizer_updates=representation,
            auxiliary_optimizer_updates=auxiliary,
            auxiliary_idm_contract=observed.get("auxiliary_idm_contract"),
            expected_optimizer_updates=2 if smoke else observed["expected_updates"],
            optimizer_evidence=optimization,
            batch_size=(2 if smoke else 128)
            if args.model in ("jepa4r", "lewm")
            else (128 if args.model == "pldm" or smoke else 256),
            artifacts=[
                ckpt / "config.json",
                ckpt / f"weights_epoch_{epoch}.pt",
                args.data_root / "audit.json",
                args.data_root / ("train_pcd.h5" if args.model == "jepa4r" else "train_rgb.h5"),
            ]
            + ([ckpt / "config.yaml"] if (ckpt / "config.yaml").exists() else []),
        )
    except Exception as error:
        from jepa4r.models.relocate.recovery import RecoveryPause

        if isinstance(error, RecoveryPause):
            raise SystemExit(75) from error
        raise
    finally:
        pl.Trainer = original_trainer
        for handle in handles:
            handle.remove()


def reload_wm(args):
    import torch
    from jepa4r.data.relocate.adroit_data import receipt

    ckpt = args.output / "checkpoints/relocate"
    state = torch.load(ckpt / f"weights_epoch_{args.epoch}.pt", map_location="cpu", weights_only=False)
    if args.model == "lejepa":
        import h5py
        from PIL import Image
        from jepa4r.baselines.relocate.train_lejepa_relocate import Encoder
        from jepa4r.policy.bc.jepa_baselines import _transforms

        model = Encoder().eval()
        other = Encoder().eval()
        for name in ("backbone", "head", "proj"):
            getattr(model, name).load_state_dict(state[name], strict=True)
            getattr(other, name).load_state_dict(state[name], strict=True)
        with h5py.File(args.data_root / "train_rgb.h5", "r") as source:
            x = _transforms()(Image.fromarray(source["pixels"][0])).unsqueeze(0)
        with torch.no_grad():
            first = model.embed(x)
            second = other.embed(x)
        pred_loss = None
    else:
        import hydra
        import stable_worldmodel as swm
        import stable_pretraining as spt
        from omegaconf import OmegaConf
        from jepa4r.data.clips import get_column_normalizer

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        cfg = OmegaConf.load(ckpt / "config.yaml")
        from jepa4r.models.world_model import _retarget  # config.json keeps LeWM's names; the classes moved

        model_config = OmegaConf.create(_retarget(json.loads((ckpt / "config.json").read_text())))
        model = hydra.utils.instantiate(model_config).eval()
        other = hydra.utils.instantiate(model_config).eval()
        model.load_state_dict(state, strict=True)
        other.load_state_dict(state, strict=True)
        source = args.data_root / ("train_pcd.h5" if args.model == "jepa4r" else "train_rgb.h5")
        cfg.data.dataset.name = str(source)
        settings = OmegaConf.to_container(cfg.data.dataset, resolve=True)
        settings.pop("name")
        dataset = swm.data.load_dataset(str(source), transform=None, **settings)
        transform = [reload_preprocessor(args.model, cfg, hydra.utils.instantiate)]
        normalize = swm.data.column_normalizer if args.model == "pldm" else get_column_normalizer
        transform.extend(
            normalize(dataset, key, key)
            for key in cfg.data.dataset.keys_to_load
            if not key.startswith(("pixels", "pcd"))
        )
        dataset.transform = spt.data.transforms.Compose(*transform)
        raw = dataset[0]
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in raw.items()}
        batch["action"] = torch.nan_to_num(batch["action"])
        recipe = cfg.wm if args.model == "pldm" else cfg
        with torch.no_grad():
            encoded = model.encode({k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()})
            encoded2 = other.encode({k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()})
            first = model.predict(
                encoded["emb"][:, : recipe.history_size], encoded["act_emb"][:, : recipe.history_size]
            )
            second = other.predict(
                encoded2["emb"][:, : recipe.history_size], encoded2["act_emb"][:, : recipe.history_size]
            )
            pred_loss = float((first - encoded["emb"][:, recipe.num_preds :]).square().mean())
        if encoded["emb"].shape[-1] != 192 or not torch.equal(encoded["emb"], encoded2["emb"]):
            raise ValueError("reload embedding contract differs")
    if not torch.equal(first, second) or not torch.isfinite(first).all():
        raise ValueError("strict reload output mismatch/nonfinite")
    receipt(
        args.output / "reload.json",
        passed=True,
        actual_exit_code=0,
        strict_reload=True,
        scope="complete exported representation model"
        if args.model == "lejepa"
        else "complete exported world-model state",
        model=args.model,
        output_shape=list(first.shape),
        prediction_loss=pred_loss,
        maximum_output_difference=0.0,
        artifacts=[ckpt / "config.json", ckpt / f"weights_epoch_{args.epoch}.pt", args.data_root / "audit.json"]
        + [args.data_root / ("train_pcd.h5" if args.model == "jepa4r" else "train_rgb.h5"), args.output / "train.json"]
        + ([ckpt / "config.yaml"] if (ckpt / "config.yaml").exists() else []),
    )


def bc(args):
    frozen_source = verify_frozen_checkpoint(args.model, args.checkpoint)
    import torch
    import numpy as np
    from jepa4r.data.relocate.adroit_data import receipt, sha
    from jepa4r.policy.bc.relocate.encoder_identity import encoder_digest, assert_encoder_unchanged
    import lewm_cortex
    import jepa_baselines
    import train_loop

    if args.output.exists():
        raise FileExistsError(args.output)
    selector = {
        "lewm": "lewm_adroit_relocate_2d",
        "jepa4r": "lewm_adroit_relocate_pcd",
        "pldm": "pldm_relocate",
        "lejepa": "lejepa_relocate",
        "visreg": "visreg_vitb",
        "vjepa": "vjepa21_vitb",
    }[args.model]
    os.environ["STABLEWM_HOME"] = str(args.output / "assets")
    for key in (
        "RELOCATE_RGB_CHECKPOINT",
        "RELOCATE_PCD_CHECKPOINT",
        "RELOCATE_PLDM_CHECKPOINT",
        "RELOCATE_LEJEPA_CHECKPOINT",
    ):
        os.environ[key] = str(args.checkpoint)
    if os.environ.get("ADROIT_VJEPA_ROOT"):
        jepa_baselines.VJEPA2_REPO = Path(os.environ["ADROIT_VJEPA_ROOT"])
    original_load = {}
    encoders = []

    def digest(model):
        return encoder_digest(model)

    for module, name in (
        (lewm_cortex, "load_lewm"),
        (lewm_cortex, "load_lewm_pcd"),
        (jepa_baselines, "load_pldm"),
        (jepa_baselines, "load_lejepa"),
        (jepa_baselines, "load_visreg"),
        (jepa_baselines, "load_vjepa21"),
    ):
        original = getattr(module, name)
        original_load[(module, name)] = original

        def load(*a, _original=original, _name=name, **kw):
            if _name in ("load_visreg", "load_vjepa21"):
                kw["checkpoint"] = str(args.checkpoint)
            else:
                kw["epoch"] = args.epoch
            result = _original(*a, **kw)
            model = result[0]
            for p in model.parameters():
                p.requires_grad_(False)
            encoders.append((model, digest(model)))
            return result

        setattr(module, name, load)
    updates = []
    rollouts = []
    observers = []
    original_adam = torch.optim.Adam
    original_paths = train_loop.sample_paths
    tail = args.entry_args[1:] if args.entry_args[:1] == ["--"] else args.entry_args
    epochs = int(next(x.split("=", 1)[1] for x in tail if x.startswith("epochs=")))
    recovery_enabled = getattr(args, "bc_recovery", False)
    bc_controller = None
    bc_identity = None
    last_bc_model = []
    old_hydra_error = os.environ.get("HYDRA_FULL_ERROR")
    bc_context = nullcontext(None)
    if recovery_enabled:
        from jepa4r.policy.bc.relocate.bc_recovery import install_bc_recovery, RecoveryPause
        from jepa4r.policy.bc.relocate.bc_caller_state import save_caller_state, load_caller_state
        from jepa4r.policy.bc.relocate import bc_caller_state, bc_recovery
        from jepa4r.policy.bc.relocate import encoder_identity

        overrides = dict(x.split("=", 1) for x in tail if "=" in x and not x.startswith("--"))
        if epochs != 1 or overrides.get("num_demos") != "1" or overrides.get("bc_kwargs.batch_size") != "128":
            raise ValueError("Native BC recovery requires its explicit one-demo one-epoch 128/72 smoke")
        demos = Path(overrides["data_dir"]) / "relocate-v0.pickle"
        weight = args.checkpoint / f"weights_epoch_{args.epoch}.pt" if args.checkpoint.is_dir() else args.checkpoint
        bc_identity = dict(
            model=dict(kind="original-EAI-BatchNormMLP", input_dim=WIDTHS[args.model], action_dim=30),
            data=dict(demos_sha256=sha(demos), conversion_receipt_sha256=sha(demos.parents[1] / "report.json")),
            config={k: v for k, v in overrides.items() if k not in ("data_dir", "hydra.run.dir", "job_name")},
            split="official-BC-first-demo-smoke",
            action_dim=30,
            encoder=dict(
                method=args.model,
                weights_sha256=sha(weight),
                epoch=args.epoch,
                config_files={
                    name: sha(args.checkpoint / name)
                    for name in ("config.json", "config.yaml")
                    if args.checkpoint.is_dir() and (args.checkpoint / name).is_file()
                },
                loader_sources=dict(
                    lewm_cortex=sha(Path(lewm_cortex.__file__)), jepa_baselines=sha(Path(jepa_baselines.__file__))
                ),
            ),
            source=dict(
                eai_train_loop_sha256=sha(Path(train_loop.__file__)),
                adapter_sha256=sha(Path(__file__)),
                caller_state_sha256=sha(Path(bc_caller_state.__file__)),
                recovery_sha256=sha(Path(bc_recovery.__file__)),
                encoder_identity_sha256=sha(Path(encoder_identity.__file__)),
            ),
        )

        def recovery_event(event, live):
            last_bc_model[:] = [live["policy"].model]
            if event["kind"] == "resume_loaded":
                if len(observers) != 1 or event["epoch_records"]:
                    raise ValueError("Native bounded BC resume observer/epoch prefix differs")
                opt = live["optimizer"]
                steps = sorted({float(s["step"]) for s in opt.state.values() if "step" in s})
                saved = load_caller_state(
                    args.bc_resume_checkpoint,
                    identity=bc_identity,
                    committed_updates=event["committed_updates"],
                    model_sha256=parameter_digest(observers[0].parameters),
                    adam_steps=steps,
                )
                observers[0].load_history_state_dict(saved["optimizer_observer"])
                if observers[0].count != event["committed_updates"]:
                    raise ValueError("BC caller observer history length differs")
                updates.extend([1] * event["committed_updates"])

        bc_context = install_bc_recovery(
            train_loop,
            bc_identity,
            args.output / "recovery",
            enabled=True,
            resume_from=args.bc_resume_checkpoint,
            pause_after_updates=args.bc_pause_after_updates,
            observer=recovery_event,
        )
        os.environ["HYDRA_FULL_ERROR"] = "1"

    def seal_caller():
        if bc_controller is None:
            return None
        if len(observers) != 1 or not last_bc_model or bc_controller.last_checkpoint is None:
            raise ValueError("BC caller has no sealed original head checkpoint")
        return save_caller_state(
            bc_controller.last_checkpoint,
            identity=bc_identity,
            state=dict(optimizer_observer=observers[0].history_state_dict()),
            committed_updates=len(bc_controller.records),
            model_sha256=parameter_digest(observers[0].parameters),
        )

    evaluation_epochs = [i for i in range(epochs) if (i > 0 and i % 5 == 0) or i == epochs - 1]
    policy_references = []
    evaluated_policies = []

    class Adam(original_adam):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            frozen_ids = {id(q) for model, _ in encoders for q in model.parameters()}
            if any(id(p) in frozen_ids for group in self.param_groups for p in group["params"]):
                raise ValueError("encoder entered BC optimizer")

            def count(opt, argv, kwargs):
                updates.append(1)

            self.register_step_post_hook(count)
            observers.append(OptimizerObserver(self))

    def sample_paths(*a, **kw):
        paths = original_paths(*a, **kw)
        for path in paths:
            actions = path["actions"]
            if actions.shape != (200, 30) or not np.isfinite(actions).all():
                raise ValueError("online action contract mismatch")
        if getattr(args, "policy_evidence", False):
            from jepa4r.policy.bc.relocate.policy_evidence import write_policy_reference

            policy = kw["policy"]
            index = len(policy_references)
            policy_references.append(
                write_policy_reference(
                    args.output / "policy_evidence",
                    f"round{index:03d}",
                    policy,
                    np.asarray(paths[0]["observations"]),
                    epoch=evaluation_epochs[index],
                    score=float(np.mean([np.sum(p["rewards"]) for p in paths])),
                )
            )
            evaluated_policies.append(policy)
        rollouts.append(len(paths))
        return paths

    torch.optim.Adam = Adam
    train_loop.sample_paths = sample_paths
    sys.argv = [
        str(ROOT / "policy/bc" / ("run_adroit_bc3d.py" if args.model == "jepa4r" else "run_bc.py")),
        *tail,
        "embedding=" + selector,
    ]
    try:
        try:
            with bc_context as bc_controller:
                runpy.run_path(sys.argv[0], run_name="__main__")
        except BaseException as error:
            if recovery_enabled and isinstance(error, RecoveryPause):
                caller = seal_caller()
                receipt(
                    args.output / "pause.json",
                    passed=True,
                    actual_exit_code=75,
                    phase="intentional_sealed_pause",
                    committed_optimizer_updates=len(bc_controller.records),
                    scope="original BC loop paused; not completed training",
                    artifacts=[bc_controller.last_checkpoint, caller],
                )
                raise SystemExit(75) from error
            raise
        caller = seal_caller()
        metrics_path = args.output / "run/metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        trajectories = int(next(x.split("=", 1)[1] for x in tail if x.startswith("eval_num_traj=")))
        demos_count = int(next(x.split("=", 1)[1] for x in tail if x.startswith("num_demos=")))
        batch_size = int(next(x.split("=", 1)[1] for x in tail if x.startswith("bc_kwargs.batch_size=")))
        expected_updates = epochs * math.ceil(demos_count * 200 / batch_size)
        summary = validate_metrics(metrics, epochs)
        if (
            len(updates) != expected_updates
            or len(rollouts) != summary["evaluation_rounds"]
            or any(n != trajectories for n in rollouts)
        ):
            raise ValueError("actual optimization or online rollout budget differs")
        if not encoders:
            raise ValueError("BC has no actual frozen encoder")
        for model, before in encoders:
            assert_encoder_unchanged(model, before)
        optimization = [observer.report() for observer in observers]
        if len(optimization) != 1 or optimization[0]["hook_count"] != len(updates):
            raise ValueError("BC Adam state/hook count mismatch")
        policy = args.output / "run/policy/iterations/best_policy.pickle"
        demos = Path(next(x.split("=", 1)[1] for x in tail if x.startswith("data_dir="))) / "relocate-v0.pickle"
        weight = args.checkpoint / f"weights_epoch_{args.epoch}.pt" if args.checkpoint.is_dir() else args.checkpoint
        replay = {}
        replay_artifacts = []
        if getattr(args, "policy_evidence", False):
            from jepa4r.policy.bc.relocate.policy_evidence import select_best_reference, verify_policy_reference

            selected = select_best_reference(policy_references)
            final_path = policy.with_name("final_policy.pickle")
            with final_path.open("xb") as stream:
                pickle.dump(evaluated_policies[-1], stream, protocol=4)
            with policy.open("rb") as stream:
                selected_policy = pickle.load(stream)
            selected_policy.model.to("cpu")
            selected_policy.model.eval()
            replay = dict(
                selected_reference=str(Path(selected["metadata_path"]).relative_to(args.output)),
                final_reference=str(Path(policy_references[-1]["metadata_path"]).relative_to(args.output)),
                selected_epoch=selected["epoch"],
                final_epoch=policy_references[-1]["epoch"],
                selected_replay=verify_policy_reference(selected_policy, selected["metadata_path"]),
            )
            replay_artifacts = [
                final_path,
                *[Path(r[k]) for r in policy_references for k in ("metadata_path", "arrays_path")],
            ]
        receipt(
            args.output / "report.json",
            passed=True,
            actual_exit_code=0,
            optimizer_updates=len(updates),
            expected_optimizer_updates=expected_updates,
            frozen_encoder_unchanged=True,
            rollouts=sum(rollouts),
            optimizer_evidence=optimization,
            frozen_encoder_sha256=[before for _, before in encoders],
            frozen_source=frozen_source,
            policy_evidence=replay,
            recovery=dict(
                enabled=True,
                checkpoint=str(bc_controller.last_checkpoint),
                caller_state=str(caller),
                batch_records=bc_controller.records,
                epoch_records=bc_controller.epoch_records,
            )
            if bc_controller is not None
            else dict(enabled=False),
            **summary,
            artifacts=[metrics_path, policy, demos, weight, demos.parents[1] / "report.json"]
            + replay_artifacts
            + (
                [ROOT.parent / "configs/policy/bc/relocate/DEPENDENCIES.json"]
                if args.model in ("visreg", "vjepa")
                else [args.checkpoint / "config.json"]
            ),
        )
    finally:
        if recovery_enabled:
            if old_hydra_error is None:
                os.environ.pop("HYDRA_FULL_ERROR", None)
            else:
                os.environ["HYDRA_FULL_ERROR"] = old_hydra_error
        torch.optim.Adam = original_adam
        train_loop.sample_paths = original_paths
        for observer in observers:
            observer.remove()
        for (module, name), original in original_load.items():
            setattr(module, name, original)


def reload_policy(args):
    import torch
    from jepa4r.data.relocate.adroit_data import receipt

    path = args.output / "run/policy/iterations/best_policy.pickle"
    with path.open("rb") as stream:
        first = pickle.load(stream)
    first.model.to("cpu")
    first.model.eval()
    x = torch.zeros((2, WIDTHS[args.model]), dtype=torch.float32)
    with torch.no_grad():
        a = first.model(x)
    with path.open("rb") as stream:
        second = pickle.load(stream)
    second.model.to("cpu")
    second.model.eval()
    with torch.no_grad():
        b = second.model(x)
    if a.shape != (2, 30) or not torch.isfinite(a).all() or not torch.equal(a, b):
        raise ValueError("saved BC head failed independent CPU reload")
    replay = {}
    extra = []
    if getattr(args, "policy_evidence", False):
        from jepa4r.policy.bc.relocate.policy_evidence import verify_policy_reference

        references = json.loads((args.output / "report.json").read_text())["policy_evidence"]
        selected = args.output / references["selected_reference"]
        final = args.output / references["final_reference"]
        final_policy_path = path.with_name("final_policy.pickle")
        with final_policy_path.open("rb") as stream:
            final_policy = pickle.load(stream)
        final_policy.model.to("cpu")
        final_policy.model.eval()
        replay = dict(
            selected=verify_policy_reference(first, selected), final=verify_policy_reference(final_policy, final)
        )
        extra = [final_policy_path, selected, final]
    receipt(
        args.output / "reload.json",
        passed=True,
        actual_exit_code=0,
        strict_reload=True,
        scope="saved BC head CPU reload",
        input_dim=WIDTHS[args.model],
        output_dim=30,
        policy_evidence=replay,
        artifacts=[path, args.output / "report.json"] + extra,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "reload-wm", "bc", "reload-policy"))
    parser.add_argument("--model", required=True, choices=tuple(WIDTHS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument(
        "--recovery", action="store_true", help="Explicit bounded original-loop full-state recovery; default off"
    )
    parser.add_argument(
        "--clip-evidence",
        action="store_true",
        help="Read original gradient clipping return and post-clip norms without changing clipping",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--pause-after-batches", type=int)
    parser.add_argument(
        "--policy-evidence",
        action="store_true",
        help="Retain real policy inputs and compare the corresponding in-memory and independently loaded states",
    )
    parser.add_argument(
        "--bc-recovery", action="store_true", help="Explicit original BC 128/72 smoke recovery; default off"
    )
    parser.add_argument("--bc-resume-checkpoint", type=Path)
    parser.add_argument("--bc-pause-after-updates", type=int)
    args, tail = parser.parse_known_args()
    args.entry_args = tail
    configure_runtime()
    if (args.resume_checkpoint is not None or args.pause_after_batches is not None) and not args.recovery:
        parser.error("recovery checkpoint/pause requires --recovery")
    if args.recovery and (
        args.stage != "train" or args.mode != "smoke" or args.model not in ("jepa4r", "lewm", "pldm", "lejepa")
    ):
        parser.error("recovery is an explicit bounded trainable-model smoke interface")
    if args.pause_after_batches is not None and args.pause_after_batches != 1:
        parser.error("the two-batch recovery gate pauses after exactly one committed batch")
    if (args.bc_resume_checkpoint is not None or args.bc_pause_after_updates is not None) and not args.bc_recovery:
        parser.error("BC checkpoint/pause requires --bc-recovery")
    if args.bc_recovery and args.stage != "bc":
        parser.error("BC recovery is available only for the BC stage")
    if args.bc_pause_after_updates is not None and args.bc_pause_after_updates != 1:
        parser.error("BC recovery pauses after exactly one committed update")
    {"train": train, "reload-wm": reload_wm, "bc": bc, "reload-policy": reload_policy}[args.stage](args)


if __name__ == "__main__":
    main()
