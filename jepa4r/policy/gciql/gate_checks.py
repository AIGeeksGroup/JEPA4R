"""Short numerical checks of the Cube workflow (a training step of each model, GCIQL resume and reload, the
encoder x input matrix), run around the existing model and update code."""

import importlib
import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback


def _tool(name):
    """One of the checking tools these checks use (lightning_compare, matrix_source, state_replay, tiny_cpu_fixture),
    kept with the Cube workflow's tests in tests/cube/; the pipeline itself never runs them."""
    tests = str(Path(__file__).resolve().parents[3] / "tests" / "cube")
    if tests not in sys.path:
        sys.path.append(tests)
    return importlib.import_module(name)


STAGES = (
    "tiny-step",
    "gciql-step",
    "gciql-cpu-reference",
    "gciql-reload",
    "compare-resume",
    "model-numerics",
    "compare-numerics",
    "matrix-forward",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pinned(record):
    path = Path(record["path"]).resolve()
    if sha(path) != record["sha256"]:
        raise ValueError("gate input/checkpoint checksum differs: " + str(path))
    return path


def build_plan(stage, config, output):
    if stage not in STAGES:
        raise ValueError("unknown Cube gate stage")
    path = Path(config).resolve()
    cfg = read(path)
    if cfg.get("mode") != "smoke":
        raise ValueError("gate runner accepts explicit smoke mode only")
    stop = cfg.get("stop_after", 2)
    if type(stop) is not int or stop not in (1, 2):
        raise ValueError("gate budget is one or two total optimizer updates")
    return dict(
        schema="cube_smoke_gate_plan_v1",
        stage=stage,
        config=str(path),
        config_sha256=sha(path),
        output=str(Path(output).resolve()),
        maximum_optimizer_updates=2,
        stop_after=stop,
        executed=False,
    )


@contextmanager
def argv(values):
    previous = sys.argv
    sys.argv = values
    try:
        yield
    finally:
        sys.argv = previous


class TinyFixedGate:
    """Feed fixed, pre-recorded augmented batches to LeJEPA Tiny's own training step."""

    def __init__(self, config, output):
        import numpy as np
        from jepa4r.utils.training_state import FixedBatchCursor

        self.config, self.output = config, Path(output)
        self.entries = config["batches"]
        if len(self.entries) < 3:
            raise ValueError("two update batches and the next batch identity are required")
        self.cursor = FixedBatchCursor([row["id"] for row in self.entries])
        with np.load(pinned(config["batch_arrays"]), allow_pickle=False) as data:
            self.arrays = {name: data[name].copy() for name in data.files}
        self.identity = config["identity"]
        if (
            config.get("model") != "lejepa"
            or self.identity.get("model") != "lejepa"
            or self.identity.get("action_dim") != 5
            or self.identity.get("split") != "train"
            or self.identity.get("data") != config["batch_arrays"]["sha256"]
        ):
            raise ValueError("Tiny fixed-batch model/action/split/input identity differs")
        self.epoch = 0

    def remaining_progress(self, progress=None):
        """Refuse completed or mismatched state before the native loop can run."""
        stop, position = self.config["stop_after"], self.cursor.position
        if type(stop) is not int or stop not in (1, 2) or position >= stop:
            raise ValueError("Tiny checkpoint has no remaining bounded update budget")
        state = self.cursor.state_dict()
        if (
            len(self.entries) <= stop
            or state["batch_ids"] != [row["id"] for row in self.entries]
            or state["next_batch_id"] != self.entries[position]["id"]
        ):
            raise ValueError("Tiny remaining entries or next batch ID differs from restored cursor")
        row = self.entries[position]
        if progress is not None and (
            any(type(progress.get(key)) is not int for key in ("step", "epoch", "batch_index"))
            or progress["step"] != position
            or progress["epoch"] != row["epoch"]
            or progress["batch_index"] != row["batch_index"]
        ):
            raise ValueError("Tiny restored progress step/epoch/batch differs from its next real batch")
        return row

    def loader(self, args):
        import torch

        owner = self
        if args.bs != 256 or args.epochs != 10:
            raise ValueError("Cube Tiny smoke retains native batch256 and ten-epoch schedule")
        actual_batch = args.bs
        if self.config.get("numerical_fixture") is not None:
            validate_config = _tool("tiny_cpu_fixture").validate_config
            actual_batch = validate_config(self.config)

        class Loader:
            dataset = range(owner.config["loader_samples"])

            def __len__(self):
                return owner.config["loader_batches_per_epoch"]

            def __iter__(self):
                while owner.cursor.position < len(owner.entries):
                    row = owner.entries[owner.cursor.position]
                    if row["epoch"] != owner.epoch:
                        break
                    value = torch.from_numpy(owner.arrays[row["array_key"]])
                    if tuple(value.shape) != (actual_batch, args.V, 3, 224, 224) or value.dtype != torch.float32:
                        raise ValueError("fixed Tiny batch must be the real FP32 augmented native input")
                    yield value

        return Loader()

    def attach(self, net, sigreg, opt, scheduler, scaler):
        import random
        import numpy as np
        import torch
        from jepa4r.utils.training_state import restore_torch_state

        self.net, self.opt, self.scheduler, self.scaler = net, opt, scheduler, scaler
        self.model = torch.nn.ModuleDict(dict(encoder=net, sigreg=sigreg))
        if self.config.get("resume_state") is not None:
            progress = restore_torch_state(
                pinned(self.config["resume_state"]),
                model=self.model,
                optimizers={"main": opt},
                schedulers={"main": scheduler},
                scaler=scaler,
                identity=self.identity,
                cursor=self.cursor,
            )
            self.remaining_progress(progress)
            return progress["epoch"]
        if self.config.get("initial_training_state") is not None:
            progress = restore_torch_state(
                pinned(self.config["initial_training_state"]),
                model=self.model,
                optimizers={"main": opt},
                schedulers={"main": scheduler},
                scaler=scaler,
                identity=self.identity,
                cursor=self.cursor,
            )
            if progress["step"] != 0 or self.cursor.position != 0:
                raise ValueError("Tiny initial reference must precede every actual update")
            self.remaining_progress(progress)
            return progress["epoch"]
        if self.config.get("initial_checkpoint") is not None:
            weights = torch.load(pinned(self.config["initial_checkpoint"]), map_location="cpu", weights_only=True)
            if set(weights) != {"backbone", "head", "proj"}:
                raise ValueError("Tiny initial native export keys differ")
            for name, value in weights.items():
                getattr(net, name).load_state_dict(value, strict=True)
        random.seed(self.config["seed"])
        np.random.seed(self.config["seed"])
        torch.manual_seed(self.config["seed"])
        self.remaining_progress()
        return 0

    def next_batch_index(self, epoch):
        row = self.remaining_progress()
        if row["epoch"] != epoch:
            raise ValueError("fixed batch epoch differs from restored progress")
        self.epoch = epoch
        return row["batch_index"]

    def autocast(self):
        from torch.amp import autocast

        if self.config["precision"] == "fp32":
            return nullcontext()
        if self.config["precision"] != "bf16":
            raise ValueError("Tiny gate precision must be explicit fp32 or bf16")
        import torch

        return autocast("cuda", dtype=torch.bfloat16)

    def before_step(self, net):
        self.before = {name: p.detach().cpu().clone().numpy() for name, p in net.named_parameters()}

    def after_step(self, epoch, step, **outputs):
        import numpy as np
        from jepa4r.utils.training_state import save_torch_state

        consumed = self.cursor.take()
        count = self.cursor.position
        arrays = {"output/" + k: v.detach().float().cpu().numpy() for k, v in outputs.items()}
        for name, p in self.net.named_parameters():
            arrays["before/" + name] = self.before[name]
            arrays["after/" + name] = p.detach().cpu().numpy()
            if p.grad is not None:
                arrays["gradient/" + name] = p.grad.detach().float().cpu().numpy()
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError("nonfinite Tiny output/gradient/parameters")
        if not any(np.any(v) for k, v in arrays.items() if k.startswith("gradient/")):
            raise ValueError("Tiny has no nonzero gradient")
        with (self.output / f"step_{count}_arrays.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        row = self.entries[count]
        checkpoint = self.output / f"step_{count}.pt"
        save_torch_state(
            checkpoint,
            model=self.model,
            optimizers={"main": self.opt},
            schedulers={"main": self.scheduler},
            scaler=self.scaler,
            identity=self.identity,
            cursor=self.cursor,
            progress=dict(step=count, epoch=row["epoch"], batch_index=row["batch_index"]),
        )
        save(
            self.output / f"step_{count}.json",
            dict(
                actual_exit_code=0,
                batch_id=consumed,
                next_batch_id=row["id"],
                epoch=epoch,
                batch_index=step,
                optimizer_steps=sorted({int(v["step"]) for v in self.opt.state.values() if "step" in v}),
                lr_after_step=[group["lr"] for group in self.opt.param_groups],
                checkpoint_sha256=sha(checkpoint),
                precision=self.config["precision"],
            ),
        )
        return count >= self.config["stop_after"]


def run_tiny(cfg, output):
    from jepa4r.baselines.cube.native import tiny_train

    gate = TinyFixedGate(cfg, output)
    previous = os.environ.get("STABLEWM_HOME")
    os.environ["STABLEWM_HOME"] = str(output / "native")
    values = [
        "cube-tiny-fixed-batch-gate",
        "--task",
        "fixed_real_batches",
        "--dataset",
        cfg["dataset"],
        "--name",
        "gate",
        "--epochs",
        "10",
        "--bs",
        "256",
        "--workers",
        str(cfg.get("workers", 1)),
    ]
    try:
        with argv(values):
            if cfg.get("numerical_fixture") is not None:
                cpu_fp32 = _tool("tiny_cpu_fixture").cpu_fp32
                with cpu_fp32(tiny_train, cfg) as metadata:
                    save(output / "NUMERICAL_FIXTURE.json", metadata)
                    tiny_train.main(gate=gate)
            else:
                tiny_train.main(gate=gate)
    finally:
        if previous is None:
            os.environ.pop("STABLEWM_HOME", None)
        else:
            os.environ["STABLEWM_HOME"] = previous
    if gate.cursor.position != cfg["stop_after"]:
        raise ValueError("Tiny fixed-batch gate did not complete its requested real updates")
    return dict(
        completed_updates=gate.cursor.position,
        next_batch_id=gate.cursor.state_dict()["next_batch_id"],
        fixed_batch_recovery_only=True,
        arbitrary_worker_queue_recovery_verified=False,
    )


def run_gciql(cfg, output):
    from .downstream.runtime import install, runtime_environment

    os.environ.update(
        runtime_environment("gciql", action="gate", cuda_visible_devices=cfg.get("cuda_visible_devices", "0"))
    )
    runtime = read(pinned(cfg["runtime_recipe"]))
    trainer, boundary = install(runtime)
    target = output / "gate"
    resume = cfg.get("resume_root")
    boundary.gate(runtime, target, stop_after=cfg["stop_after"], resume_root=resume)
    return read(target / "report.json")


def reload_gciql(cfg, output):
    from .downstream.runtime import install, runtime_environment

    os.environ.update(runtime_environment("gciql", action="reload"))
    runtime = read(pinned(cfg["runtime_recipe"]))
    _, boundary = install(runtime)
    target = output / "reload.json"
    boundary.reload_gate(
        runtime, Path(cfg["gate_root"]), target, cpu_reference=cfg.get("cpu_reference"), require_cpu_reference=True
    )
    return read(target)


def cpu_reference_gciql(cfg, output):
    from .downstream.runtime import install, runtime_environment

    os.environ.update(runtime_environment("gciql", action="reload"))
    runtime = read(pinned(cfg["runtime_recipe"]))
    trainer, boundary = install(runtime)
    from .policy_reference import write_cpu_reference

    return write_cpu_reference(runtime, Path(cfg["gate_root"]), output, trainer, boundary, cfg["runtime_recipe"])


def _state_cpu_arrays(item):
    """Preserve the complete tree while comparing stored Torch RNG on CPU."""
    if hasattr(item, "detach") and hasattr(item, "cpu"):
        return item.detach().cpu().numpy()
    if isinstance(item, dict):
        return {key: _state_cpu_arrays(value) for key, value in item.items()}
    if isinstance(item, (tuple, list)):
        return type(item)(_state_cpu_arrays(value) for value in item)
    return item


def _gciql_sidecar(cfg, role):
    """Verify the GCIQL checkpoint (its manifest and checksums) before reading its RNG state."""
    import pickle

    try:
        path = pinned(cfg[role + "_sidecar"])
        manifest_path = pinned(cfg[role + "_checkpoint_manifest"])
    except KeyError as error:
        raise ValueError("both complete GCIQL sidecar/manifest bindings are required") from error
    if path.name != "run_state.pkl" or manifest_path.parent != path.parent:
        raise ValueError("GCIQL sidecar and checkpoint transaction directories differ")
    manifest = read(manifest_path)
    if type(manifest.get("counter")) is not int or manifest["counter"] != 2:
        raise ValueError("GCIQL recovery comparison requires the completed two-update boundary")
    agent_path = path.parent / "params_2.pkl"
    if manifest.get("sidecar_sha256") != sha(path) or manifest.get("agent_sha256") != sha(agent_path):
        raise ValueError("GCIQL checkpoint transaction agent/sidecar checksum differs")
    # The pickle was written by this run and its checksum was checked above; it is not a downloaded file.
    with path.open("rb") as stream:
        state = pickle.load(stream)
    required = {"binding", "rng", "budget_total", "events", "network_step", "adam_count"}
    if not isinstance(state, dict) or not required <= set(state):
        raise ValueError("GCIQL sidecar is missing complete training state")
    if set(state["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("GCIQL sidecar is missing a host RNG field")
    for name, expected in [("network_step", 3), ("adam_count", 2), ("budget_total", 1000000)]:
        if type(state[name]) is not int or state[name] != expected:
            raise ValueError("GCIQL sidecar update/budget counter differs: " + name)
    events = state["events"]
    for name, expected in [
        ("committed_updates", 2),
        ("validation_count", 0),
        ("last_validation", 0),
        ("next_update", 3),
    ]:
        if type(events.get(name)) is not int or events[name] != expected:
            raise ValueError("GCIQL sidecar committed event differs: " + name)
    if events.get("checkpoint_event_complete") is not True or events.get("next_stage") != "sample_train":
        raise ValueError("GCIQL sidecar next event is not the committed sample boundary")
    binding = state["binding"]
    if not {"config", "source_bindings", "code"} <= set(binding):
        raise ValueError("GCIQL sidecar lacks original data/config/source bindings")
    recipe = binding["config"]
    if recipe.get("total_updates") != 1000000 or recipe.get("agent_config", {}).get("lr") != 3e-4:
        raise ValueError("GCIQL sidecar does not use the original constant-Adam recipe")
    if state.get("scheduler") is not None or state.get("amp_scaler") is not None:
        raise ValueError("scheduler/scaler state is unsupported by the frozen JAX FP32 recipe")
    return _state_cpu_arrays(state), dict(
        manifest_sha256=sha(manifest_path), sidecar_sha256=sha(path), agent_sha256=sha(agent_path), counter=2
    )


def compare_resume(cfg):
    import numpy as np
    from jepa4r.policy.gciql.gates.arrays import compare_array_maps, compare_nested

    if cfg["kind"] == "lightning":
        compare = _tool("lightning_compare").compare
        expected = read(pinned(cfg["expected_identity"]))["identity"]
        return compare(cfg["actual_state"], cfg["reference_state"], expected)
    if cfg["kind"] == "gciql":
        actual_sidecar, actual_checkpoint = _gciql_sidecar(cfg, "actual")
        reference_sidecar, reference_checkpoint = _gciql_sidecar(cfg, "reference")
        with (
            np.load(pinned(cfg["actual_state"]), allow_pickle=False) as a,
            np.load(pinned(cfg["reference_state"]), allow_pickle=False) as b,
        ):
            result = compare_array_maps(dict(a), dict(b), exact=True)
        for field in ("next_batch", "next_ids"):
            with (
                np.load(pinned(cfg["actual_" + field]), allow_pickle=False) as a,
                np.load(pinned(cfg["reference_" + field]), allow_pickle=False) as b,
            ):
                result[field] = compare_array_maps(dict(a), dict(b), exact=True)
                result["passed"] &= result[field]["passed"]
        result["sidecar"] = compare_nested(actual_sidecar, reference_sidecar, exact=True)
        result["passed"] &= result["sidecar"]["passed"]
        result["checkpoint_transactions"] = dict(actual=actual_checkpoint, reference=reference_checkpoint)
        result["state_applicability"] = dict(
            scheduler="not_applicable_constant_optax_adam", amp_scaler="not_applicable_jax_fp32"
        )
        return result
    if cfg["kind"] != "torch":
        raise ValueError("resume comparison kind must be torch or gciql")
    import torch

    def load(record):
        path = pinned(record)
        receipt = read(path.with_suffix(path.suffix + ".json"))
        if receipt["sha256"] != sha(path) or receipt["actual_exit_code"] != 0:
            raise ValueError("unaccepted training state")
        value = torch.load(path, map_location="cpu", weights_only=False)
        return _state_cpu_arrays(value)

    return compare_nested(load(cfg["actual_state"]), load(cfg["reference_state"]), exact=True)


def observe_pldm_result(model, captured):
    """Read full native encoding/prediction tensors without another forward."""
    original = model.forward_result

    def forward_result(*args, **kwargs):
        value = original(*args, **kwargs)
        captured["encoder_full"] = value.backbone_output.encodings
        captured["prediction"] = value.pred_output.predictions
        captured["actions_used"] = value.actions
        return value

    model.forward_result = forward_result


def model_numerics(cfg, output):
    """Observe the original LeWM/JEPA4R/PLDM forward and one positive FP32 update."""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from types import SimpleNamespace
    from jepa4r.models.cube.training_runtime import pinned_native, code_root

    code = code_root().resolve()
    expected_root = Path(cfg["source_root"]).resolve()
    if (expected_root / "lewm_pcd").is_dir():  # an earlier copy of this code in its old layout; else this repository
        expected_root = expected_root / "lewm_pcd"
    if code != expected_root:
        raise ValueError("actual imported model source root differs from frozen expected root")
    source_files = {}
    for name, expected_sha in cfg["expected_source_files"].items():
        path = (code / name).resolve()
        if not path.is_relative_to(code) or sha(path) != expected_sha:
            raise ValueError("actual numerical producer source file differs: " + name)
        source_files[str(path)] = expected_sha
    method = cfg["model"]
    device = cfg.get("device", "cpu")
    torch.manual_seed(cfg["seed"])
    with np.load(pinned(cfg["batch_arrays"]), allow_pickle=False) as data:
        batch = {k: torch.from_numpy(data[k].copy()).to(device) for k in data.files}
    captured = {}
    if method in ("lewm_pcd", "lewm_rgb"):
        recipe = OmegaConf.create(read(pinned(cfg["model_config"])))
        with pinned_native() as native:
            if Path(native.__file__).resolve() not in (
                code / "bench/ogbench/native/lewm_train.py",
                code / "jepa4r/models/world_model.py",
            ):
                raise ValueError("numerical gate imported another native model implementation")
            model = native.hydra.utils.instantiate(recipe.model).to(device)
            model.load_state_dict(
                torch.load(pinned(cfg["checkpoint"]), map_location=device, weights_only=True), strict=True
            )
            model.train(cfg.get("training", True))
            original_predict = model.predict

            def predict(*args, **kwargs):
                value = original_predict(*args, **kwargs)
                captured["prediction"] = value
                return value

            model.predict = predict
            receiver = SimpleNamespace(
                model=model, sigreg=native.SIGReg(**recipe.loss.sigreg.kwargs).to(device), log_dict=lambda *a, **k: None
            )
            optimizer_spec = dict(recipe.optimizer)
            production_spec = {
                "model_opt": dict(
                    modules="model",
                    optimizer=optimizer_spec,
                    scheduler={"type": "LinearWarmupCosineAnnealingLR"},
                    interval="epoch",
                )
            }
            from functools import partial

            owner = native.spt.Module(
                model=model,
                sigreg=receiver.sigreg,
                forward=partial(native.lejepa_forward, cfg=recipe),
                optim=production_spec,
            )
            groups, named_groups, _ = owner._collect_parameters_by_optimizer_groups(list(production_spec.items()))
            opt = native.spt.optim.create_optimizer(
                groups["model_opt"], optimizer_spec, named_params=named_groups["model_opt"]
            )
            optimizer_factory = Path(inspect.getfile(native.spt.optim.create_optimizer)).resolve()
            source_files[str(optimizer_factory)] = sha(optimizer_factory)
            production_module = Path(inspect.getfile(native.spt.Module)).resolve()
            source_files[str(production_module)] = sha(production_module)
            torch.manual_seed(cfg["seed"])
            outputs = native.lejepa_forward(receiver, batch, "gate", recipe)
        clip = float(recipe.trainer.get("gradient_clip_val", 1.0))
    elif method == "pldm":
        from jepa4r.baselines.cube.pldm import method as implementation

        model = implementation.build_model(
            str(Path(implementation.__file__).resolve().parents[4] / "third_party/pldm")
        ).to(device)
        model.load_state_dict(
            torch.load(pinned(cfg["checkpoint"]), map_location=device, weights_only=True), strict=True
        )
        model.train(cfg.get("training", True))
        opt = implementation.build_optimizer(model)
        schedule = implementation.build_scheduler(opt, epochs=10, batch_steps=cfg["train_batches"], batch_size=128)
        schedule.adjust_learning_rate(cfg["schedule_step"])
        observe_pldm_result(model, captured)
        torch.manual_seed(cfg["seed"])
        outputs = model(batch["pixels"], batch["actions"])
        clip = None
    else:
        raise ValueError("model-numerics supports the original LeWM/JEPA4R/PLDM implementations")
    if not all(group["lr"] > 0 for group in opt.param_groups):
        raise ValueError("M3 requires a positive learning-rate update")
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    parameter_groups = []
    for group in opt.param_groups:
        parameter_groups.append(
            dict(
                parameters=[parameter_names[id(p)] for p in group["params"]],
                options={k: v for k, v in group.items() if k != "params"},
            )
        )
    before = {k: p.detach().cpu().clone().numpy() for k, p in model.named_parameters()}
    opt.zero_grad()
    outputs["loss"].backward()
    unclipped = {
        k: p.grad.detach().float().cpu().clone().numpy() for k, p in model.named_parameters() if p.grad is not None
    }
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    arrays = {
        "output/" + k: v.detach().float().cpu().numpy()
        for k, v in {**outputs, **captured}.items()
        if isinstance(v, torch.Tensor)
    }
    arrays.update({"gradient_unclipped/" + k: v for k, v in unclipped.items()})
    for k, p in model.named_parameters():
        arrays["before/" + k] = before[k]
        if p.grad is not None:
            arrays["gradient/" + k] = p.grad.detach().float().cpu().numpy()
    opt.step()
    for k, p in model.named_parameters():
        arrays["after/" + k] = p.detach().cpu().numpy()
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("nonfinite model numerical gate")
    if not any(np.any(value) for value in unclipped.values()) or not any(
        np.any(arrays["after/" + name] != value) for name, value in before.items()
    ):
        raise ValueError("M3 requires nonzero gradients and actual parameter changes")
    with (output / "numerics.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    return dict(
        optimizer_updates=1,
        lr=[g["lr"] for g in opt.param_groups],
        precision="fp32",
        arrays_sha256=sha(output / "numerics.npz"),
        named_arrays=len(arrays),
        source_revision=cfg["source_revision"],
        checkpoint_sha256=cfg["checkpoint"]["sha256"],
        batch_sha256=cfg["batch_arrays"]["sha256"],
        seed=cfg["seed"],
        device=device,
        actual_source_root=str(code.parent),
        actual_source_files=source_files,
        optimizer_class=type(opt).__module__ + "." + type(opt).__name__,
        optimizer_groups=parameter_groups,
        optimizer_state_initial="fresh original production factory",
        scheduler_scope="fixed positive-LR M3 update; production schedule is separately checked by T1/T4",
    )


def _matrix_target_inputs(case, runtime):
    """Check a case's inputs (cube count N, states, goals, camera) and, optionally, each input against the cache rows
    it came from."""
    import numpy as np

    with np.load(pinned(case["inputs"]), allow_pickle=False) as data:
        raw = {key: data[key].copy() for key in data.files}
    names = {"current_visual", "goal_visual", "current_state", "goal_state"}
    if not names.issubset(raw):
        raise ValueError("matrix target needs current/goal visual/state arrays")
    for name in ("current_state", "goal_state"):
        value = raw[name]
        if value.shape != (19 + 9 * runtime["cube_count"],) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("matrix target N/state shape/dtype/finite differs")
    pcd = runtime["model"] == "lewm_pcd"
    for name in ("current_visual", "goal_visual"):
        value = raw[name]
        if (
            value.shape != ((1200, 6) if pcd else (224, 224, 3))
            or value.dtype != (np.float32 if pcd else np.uint8)
            or not np.isfinite(value).all()
        ):
            raise ValueError("matrix target actual RGB/PCD shape/dtype/finite differs")
    pin = case.get("input_provenance")
    resolved = None
    if case.get("source_encoder_config") is not None and pin is None:
        raise ValueError("shared encoder requires independent target input provenance")
    if pin is not None:
        descriptor = read(pinned(pin))
        if (
            descriptor.get("schema") != "cube_matrix_target_inputs_v1"
            or type(descriptor.get("cube_count")) is not int
            or descriptor["cube_count"] != runtime["cube_count"]
            or descriptor.get("camera") != runtime["camera"]
            or descriptor.get("inputs") != case["inputs"]
            or set(descriptor.get("rows", {})) != names
        ):
            raise ValueError("matrix target provenance Cube/camera/input binding differs")
        producer = read(pinned(descriptor["producer_receipt"]))
        if (
            type(producer.get("actual_exit_code")) is not int
            or producer["actual_exit_code"] != 0
            or type(producer.get("cube_count")) is not int
            or producer["cube_count"] != runtime["cube_count"]
            or producer.get("camera") != runtime["camera"]
            or producer.get("raw_inputs") != descriptor["source_arrays"]
            or not producer.get("source_binding")
        ):
            raise ValueError("matrix target actual producer/source/camera differs")
        for path, digest in producer["source_binding"].items():
            pinned(dict(path=path, sha256=digest))
        with np.load(pinned(descriptor["source_arrays"]), allow_pickle=False) as source:
            selectors = case.get("input_selectors", {})
            if set(selectors) != {"current", "goal"} or "raw_row_ids" not in source:
                raise ValueError("matrix case needs declared physical row selectors and canonical source row IDs")
            ids = source["raw_row_ids"]
            if (
                ids.ndim != 1
                or ids.dtype != np.int64
                or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0)
                or producer.get("raw_rows") != ids.tolist()
            ):
                raise ValueError("matrix canonical source row map differs from the actual producer")
            resolved = {}
            for phase in ("current", "goal"):
                visual, state = (descriptor["rows"][phase + suffix] for suffix in ("_visual", "_state"))
                index = visual.get("row")
                selector = selectors[phase]
                if (
                    not isinstance(selector, dict)
                    or set(selector) != {"raw_row_id"}
                    or type(selector["raw_row_id"]) is not int
                    or visual.get("row") != state.get("row")
                    or type(index) is not int
                    or not 0 <= index < len(ids)
                    or int(ids[index]) != selector["raw_row_id"]
                ):
                    raise ValueError("matrix phase visual/state physical row or declared selector differs")
                resolved[phase] = int(ids[index])
            for name in names:
                field = descriptor["rows"][name]
                index = field.get("row")
                if (
                    type(index) is not int
                    or index < 0
                    or field.get("key") not in source
                    or len(source[field["key"]]) != len(ids)
                    or index >= len(source[field["key"]])
                ):
                    raise ValueError("matrix target source row/key differs")
                expected = source[field["key"]][index]
                if (
                    raw[name].dtype != expected.dtype
                    or raw[name].shape != expected.shape
                    or raw[name].tobytes() != expected.tobytes()
                ):
                    raise ValueError("matrix target row differs from complete actual source array")
    return raw, dict(
        cube_count=runtime["cube_count"], inputs=case["inputs"], provenance=pin, resolved_physical_rows=resolved
    )


def _matrix_policy_config(case, runtime):
    """Check that a case's policy config and checkpoint belong to the run that produced them (model, cubes, inputs)."""
    from .workflow_downstream import group_contract

    cfg = read(pinned(case["policy_config"]))
    model, cubes = runtime["model"], runtime["cube_count"]
    identity = dict(model=model, method=model, role=f"{model}_cube{cubes}", cube_count=cubes)
    if (
        any(cfg.get(k) != v for k, v in identity.items())
        or cfg.get("input_contract") != group_contract(model, cubes)["input_contract"]
        or cfg.get("encoder_binding") != runtime["encoder_binding"]
        or type(cfg.get("seed")) is not int
        or cfg["seed"] != 0
    ):
        raise ValueError("matrix policy model/method/role/encoder/config identity differs")
    origin_pin = case.get("policy_origin")
    if origin_pin is None:
        raise ValueError("matrix policy requires its actual producer origin")
    origin = read(pinned(origin_pin))
    if (
        origin.get("schema") != "cube_M2_policy_origin_v1"
        or any(origin.get(k) != v for k, v in identity.items())
        or origin.get("policy_config") != case["policy_config"]
        or origin.get("checkpoint") != case["policy_checkpoint"]
        or origin.get("encoder_binding") != runtime["encoder_binding"]
        or origin.get("inputs") != case["inputs"]
    ):
        raise ValueError("matrix policy producer origin role/config/checkpoint differs")
    freeze_pin = origin["producer_reference_freeze"]
    freeze = read(pinned(freeze_pin))
    if type(freeze.get("actual_exit_code")) is not int or freeze["actual_exit_code"] != 0:
        raise ValueError("matrix policy source producer did not seal a successful reference")
    checkpoint = case["policy_checkpoint"]
    try:
        checkpoint_name = (
            (Path(checkpoint["path"]) / f"params_{checkpoint['counter']}.pkl")
            .relative_to(Path(freeze_pin["path"]).parent)
            .as_posix()
        )
    except ValueError as error:
        raise ValueError("matrix policy checkpoint is outside its actual producer reference") from error
    if checkpoint_name not in freeze["files"] or freeze["files"][checkpoint_name]["sha256"] != checkpoint["sha256"]:
        raise ValueError("matrix policy checkpoint was not sealed by its actual producer")

    def producer_file(key):
        record = origin[key]
        try:
            relative = Path(record["path"]).relative_to(Path(freeze_pin["path"]).parent).as_posix()
        except ValueError as error:
            raise ValueError("matrix policy producer record escapes its frozen reference") from error
        if relative not in freeze["files"] or freeze["files"][relative]["sha256"] != record["sha256"]:
            raise ValueError("matrix policy producer record is not bound by its original freeze")
        return read(pinned(record))

    ledger = producer_file("producer_cases")
    result = producer_file("producer_result")
    observer = producer_file("producer_observer")
    if (
        type(result.get("actual_exit_code")) is not int
        or result["actual_exit_code"] != 0
        or type(observer.get("actual_exit_code")) is not int
        or observer["actual_exit_code"] != 0
        or observer.get("observer") != freeze["producer"]
    ):
        raise ValueError("matrix policy actual producer result/source differs")
    pinned(freeze["producer"])
    source_cases = [row for row in ledger["cases"] if (row["model"], row["cube_count"]) == (model, cubes)]
    source_results = [row for row in result["result"]["cases"] if (row["model"], row["cube_count"]) == (model, cubes)]
    if len(source_cases) != 1 or len(source_results) != 1:
        raise ValueError("matrix policy actual producer case must be unique")
    source_case, produced = source_cases[0], source_results[0]
    if (
        source_case["policy_checkpoint"] != case["policy_checkpoint"]
        or source_case["inputs"] != case["inputs"]
        or produced["policy_sha256"] != case["policy_checkpoint"]["sha256"]
        or produced["encoder_sha256"] != runtime["encoder_binding"]["checkpoint_sha256"]
        or produced["target_inputs"]["inputs"] != case["inputs"]
        or cfg.get("source_policy_config") != source_case["policy_config"]
    ):
        raise ValueError("matrix policy checkpoint/config/input is not from this actual producer case")
    source_runtime_pin = source_case.get("source_encoder_config", source_case.get("downstream_config"))
    source_runtime = read(pinned(source_runtime_pin))
    if source_runtime.get("encoder_binding") != runtime["encoder_binding"]:
        raise ValueError("matrix policy producer used a different actual encoder binding")
    original_cfg = read(pinned(source_case["policy_config"]))
    metadata_keys = {"model", "method", "role", "encoder_binding", "source_policy_config"}
    if any(cfg.get(k) != value for k, value in original_cfg.items()) or set(cfg) - set(original_cfg) - metadata_keys:
        raise ValueError("matrix policy effective recipe differs from its original producer config")
    return cfg, dict(
        passed=True,
        identity=identity,
        origin=origin_pin,
        producer_reference_freeze=freeze_pin,
        actual_producer_policy_config=source_case["policy_config"],
    )


def _matrix_restore_policy(template, checkpoint, restore_agent, *, to_state_dict=None):
    """Observe every original policy leaf; never change the production restore."""
    import pickle

    flatten_state, state_from_arrays = _tool("state_replay").flatten_state, _tool("state_replay").state_from_arrays
    from jepa4r.policy.gciql.gates.arrays import compare_array_maps

    if to_state_dict is None:
        from flax.serialization import to_state_dict
    path = pinned(
        dict(path=str(Path(checkpoint["path"]) / f"params_{checkpoint['counter']}.pkl"), sha256=checkpoint["sha256"])
    )
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or set(payload) != {"agent"}:
        raise ValueError("matrix saved policy envelope differs")
    expected, saved = to_state_dict(template), payload["agent"]

    def structure(left, right):
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError("matrix saved policy complete key set differs")
            for key in left:
                structure(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            if type(right) is not type(left) or len(left) != len(right):
                raise ValueError("matrix saved policy sequence differs")
            for a, b in zip(left, right):
                structure(a, b)
        elif isinstance(right, (dict, list, tuple)):
            raise ValueError("matrix saved policy leaf/container differs")

    structure(expected, saved)
    saved_arrays = flatten_state(saved)
    _, mapping = state_from_arrays(expected, saved_arrays)
    # Keep the real upstream loader as the only agent restore operation.
    agent = restore_agent(template, str(path.parent), checkpoint["counter"])
    pinned(dict(path=str(path), sha256=checkpoint["sha256"]))
    restored = to_state_dict(agent)
    structure(saved, restored)
    restored_arrays = flatten_state(restored)
    comparison = compare_array_maps(restored_arrays, saved_arrays, exact=True)
    if not comparison["passed"]:
        raise ValueError("matrix restored policy full state differs from saved state")
    return (
        agent,
        restored_arrays,
        dict(
            passed=True,
            full_numeric_leaves=len(mapping),
            integer_dtype_allowances=[row for row in mapping if row["endpoint_dtype"] != row["template_dtype"]],
            leaf_mapping=mapping,
            comparison=comparison,
        ),
    )


def matrix_forward(cfg, output):
    """All 24 forwards (encoder -> inputs -> policy) of the model matrix on real inputs, loading each weight once."""
    import numpy as np
    from .downstream.runtime import runtime_environment

    os.environ.update(runtime_environment("native-sr"))
    from . import workflow_downstream as workflow

    select_source_assets = _tool("matrix_source").select_source_assets
    from .downstream._source import install_impls

    install_impls()
    from .downstream.sampler import create_agent
    from .downstream.trainer import agent_config, model, restore_agent, jax
    from .downstream.vendor.cube_inputs import CubeInputs

    expected = {(m, n) for m in workflow.MODELS for n in (1, 2, 3, 4)}
    identities = [(row["model"], row["cube_count"]) for row in cfg["cases"]]
    if len(identities) != 24 or set(identities) != expected:
        raise ValueError("matrix must contain each of the 24 Cube/model identities once")
    encoders, results = {}, []
    for case in cfg["cases"]:
        runtime = read(pinned(case["downstream_config"]))
        identity = runtime["model"], runtime["cube_count"]
        if identity != (case["model"], case["cube_count"]) or runtime["mode"] != "smoke":
            raise ValueError("matrix case/config identity differs")
        workflow.validate_config(runtime)
        raw, input_evidence = _matrix_target_inputs(case, runtime)
        if input_evidence["provenance"] is None:
            raise ValueError("full M2 matrix needs actual physical input provenance for every case")
        policy_cfg, policy_origin = _matrix_policy_config(case, runtime)
        selected = select_source_assets(case, runtime)
        source = selected["source_runtime"]
        # The complete factory/preprocessor binding also defines encoder reuse.
        key = (
            source["model"],
            hashlib.sha256(json.dumps(source["encoder_binding"], sort_keys=True).encode()).hexdigest(),
        )
        if key not in encoders:
            encoders[key] = workflow._encoder(source)
        encoder = encoders[key]
        width = workflow.MODELS[runtime["model"]][1]
        latent = encoder.encode(raw["current_visual"])
        goal_latent = encoder.encode(raw["goal_visual"])
        seam = CubeInputs(cubes=runtime["cube_count"], latent_dim=width)
        current = seam.current(latent[None], raw["current_state"][None])[0]
        goal = seam.goal(goal_latent[None], raw["goal_state"][None])[0]
        batch = dict(observations=current[None], actor_goals=goal[None], actions=np.zeros((1, 5), np.float32))
        template = create_agent(model.GCIQLAgent, policy_cfg["seed"], batch, agent_config(policy_cfg))
        agent, policy_arrays, policy_load = _matrix_restore_policy(template, case["policy_checkpoint"], restore_agent)
        action = np.asarray(agent.sample_actions(current, goal, seed=jax.random.PRNGKey(0), temperature=0))
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError("matrix policy action shape/finite failed")
        name = f"{identity[0]}_cube{identity[1]}"
        with (output / (name + ".npz")).open("xb") as stream:
            np.savez(stream, latent=latent, goal_latent=goal_latent, current=current, goal=goal, action=action)
        with (output / (name + "_policy_state.npz")).open("xb") as stream:
            np.savez_compressed(stream, **policy_arrays)
        results.append(
            dict(
                model=identity[0],
                cube_count=identity[1],
                arrays_sha256=sha(output / (name + ".npz")),
                encoder_sha256=source["encoder_binding"]["checkpoint_sha256"],
                encoder_source=selected["provenance"],
                target_inputs=input_evidence,
                policy_origin=policy_origin,
                policy_load=policy_load,
                policy_state_sha256=sha(output / (name + "_policy_state.npz")),
                policy_sha256=case["policy_checkpoint"]["sha256"],
            )
        )
    return dict(real_forward_cases=24, encoder_loads=len(encoders), cases=results)


def compare_numerics(cfg):
    import numpy as np
    from jepa4r.policy.gciql.gates.arrays import compare_array_maps, compare_parameter_updates

    with (
        np.load(pinned(cfg["actual"]), allow_pickle=False) as a,
        np.load(pinned(cfg["reference"]), allow_pickle=False) as b,
    ):
        actual, reference = dict(a), dict(b)
    result = compare_array_maps(actual, reference)

    def select(arrays, prefix):
        return {k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)}

    result["parameter_updates"] = compare_parameter_updates(
        select(actual, "before/"), select(actual, "after/"), select(reference, "before/"), select(reference, "after/")
    )
    result["passed"] &= result["parameter_updates"]["passed"]
    return result


def execute(plan):
    if sha(plan["config"]) != plan["config_sha256"]:
        raise ValueError("gate config changed after planning")
    cfg = read(plan["config"])
    cfg.setdefault("stop_after", 2)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    save(output / "PLAN.json", plan)
    started = datetime.now(timezone.utc).isoformat()
    code, result = 1, {}
    with (
        (output / "stdout_stderr.log").open("x", encoding="utf8") as stream,
        redirect_stdout(stream),
        redirect_stderr(stream),
    ):
        try:
            stage = plan["stage"]
            result = (
                run_tiny(cfg, output)
                if stage == "tiny-step"
                else run_gciql(cfg, output)
                if stage == "gciql-step"
                else cpu_reference_gciql(cfg, output)
                if stage == "gciql-cpu-reference"
                else reload_gciql(cfg, output)
                if stage == "gciql-reload"
                else compare_resume(cfg)
                if stage == "compare-resume"
                else model_numerics(cfg, output)
                if stage == "model-numerics"
                else matrix_forward(cfg, output)
                if stage == "matrix-forward"
                else compare_numerics(cfg)
            )
            if result.get("passed", True) is not True:
                raise ValueError("full-array gate comparison failed")
            code = 0
        except BaseException:
            result["error"] = traceback.format_exc()
            traceback.print_exc()
    save(
        output / "RESULT.json",
        dict(
            actual_exit_code=code,
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            producer_sha256=sha(__file__),
            result=result,
        ),
    )
    (output / "actual_exit_code").write_text(str(code) + "\n")
    return code


def main(argv_values=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--source-root", type=Path, help="direct-script observer: import the frozen source package here"
    )
    args = parser.parse_args(argv_values)
    if args.source_root is None:
        from jepa4r.models.cube.training_runtime import (
            build_environment as _source_environment,
            code_root as _code_root,
        )

        sys.path[:0] = _source_environment(_code_root().parent, args.output)["PYTHONPATH"].split(os.pathsep)
    plan = build_plan(args.stage, args.config, args.output)
    print(json.dumps(plan, indent=2))
    return execute(plan) if args.execute else 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        # Run as a script: with --source-root the checks import an earlier copy of this
        # code (to compare two versions), else this repository's package.
        if "--source-root" in sys.argv:  # the earlier copy, imported by its own package name
            source = Path(sys.argv[sys.argv.index("--source-root") + 1]).resolve()
            sys.path.insert(0, str(source / "lewm_pcd"))
            importlib.import_module("bench.ogbench")  # the package of the relative imports below
            from bench.ogbench.training_runtime import build_environment as _source_environment

            _run_output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
            sys.path[:0] = _source_environment(source, _run_output)["PYTHONPATH"].split(os.pathsep)
            __package__ = "bench.ogbench"
        else:  # this package in this repository
            source = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(source))
            importlib.import_module("jepa4r.policy.gciql")  # the package of the relative imports below
            from jepa4r.models.cube.training_runtime import build_environment as _source_environment

            _run_output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
            sys.path[:0] = _source_environment(source, _run_output)["PYTHONPATH"].split(os.pathsep)
            __package__ = "jepa4r.policy.gciql"
    raise SystemExit(main())
