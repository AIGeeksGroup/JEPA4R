"""Numerical-only device routing for the original Tiny factories and main loop."""

from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
from pathlib import Path


def validate_config(config):
    expected = dict(schema="cube_tiny_cpu_fp32_M3_v1", role="M3", configured_batch_size=256, actual_fixed_batch_size=2)
    recovery = dict(
        schema="cube_tiny_cpu_fp32_fixed_batch_v2", role="T4", configured_batch_size=256, actual_fixed_batch_size=2
    )
    is_recovery = config.get("numerical_fixture") == recovery
    if (
        (config.get("numerical_fixture") != expected and not is_recovery)
        or config.get("mode") != "smoke"
        or config.get("model") != "lejepa"
        or config.get("precision") != "fp32"
        or config.get("stop_after") != (2 if is_recovery else 1)
        or config.get("seed") != 0
    ):
        raise ValueError("CPU fixture requires explicit bounded FP32 M3/T4, configured256 and actual fixedB2")
    if is_recovery:
        _continuation(config)
    return 2


def _continuation(config):
    source = config.get("continuation_source", {})
    required = {
        "recipe",
        "first_state",
        "first_receipt",
        "initial_state",
        "first_producer",
        "source_freeze",
        "accepted_case",
    }
    if set(source) != required:
        raise ValueError("CPU T4 needs immutable original recipe/checkpoint provenance")

    def pinned(record):
        path = Path(record["path"])
        h = hashlib.sha256()
        with path.open("rb") as f:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        if h.hexdigest() != record["sha256"]:
            raise ValueError("CPU T4 original provenance file changed")
        return path

    paths = {k: pinned(v) for k, v in source.items()}
    data = {k: json.loads(paths[k].read_text()) for k in required - {"first_state", "initial_state"}}
    recipe, first = data["recipe"], data["first_receipt"]
    expected = dict(
        model="lejepa",
        seed=0,
        epochs=10,
        batch_size=256,
        views=4,
        lr=0.002,
        lamb=0.02,
        proj_dim=16,
        precision="fp32",
        loader_samples=400,
        loader_batches_per_epoch=1,
        configured_batch_size=256,
        actual_fixed_batch_size=2,
    )
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError("CPU T4 changed the original FP32 training recipe")
    if (
        first.get("actual_exit_code") != 0
        or first["progress"]["step"] != 1
        or first["cursor"]["position"] != 1
        or first["sha256"] != source["first_state"]["sha256"]
        or first["identity"] != config["identity"]
        or config["identity"]["config"] != source["recipe"]["sha256"]
        or config["batch_arrays"]["sha256"] != config["identity"]["data"]
        or first["cursor"]["batch_ids"] != [r["id"] for r in config["batches"]]
        or (config["loader_samples"], config["loader_batches_per_epoch"]) != (400, 1)
    ):
        raise ValueError("CPU T4 first-step identity/input/cursor differs")
    producer = data["first_producer"]
    accepted = data["accepted_case"]
    if (
        producer.get("actual_exit_code") != 0
        or producer["result"].get("completed_updates") != 1
        or producer["producer_sha256"] != data["source_freeze"]["files"]["lewm_pcd/bench/ogbench/gate_checks.py"]
        or accepted.get("actual_exit_code") != 0
        or not accepted["comparison"]["passed"]
        or not accepted["state_comparison"]["passed"]
    ):
        raise ValueError("CPU T4 needs the actual accepted first-step producer/source record")
    if config.get("resume_state") is not None:
        selected = config["resume_state"]
        wanted = source["first_state"]
        if config.get("initial_training_state") is not None:
            raise ValueError("CPU T4 start boundary is ambiguous")
    else:
        selected = config.get("initial_training_state")
        wanted = source["initial_state"]
    if selected != wanted:
        raise ValueError("CPU T4 must reuse the original initial0 or sealed first1 state")


def _device(value):
    if isinstance(value, str) and value in ("cuda", "cuda:0"):
        return "cpu"
    if type(value).__name__ == "device" and getattr(value, "type", None) == "cuda":
        return "cpu"
    return value


@contextmanager
def cpu_fp32(module, config):
    batch = validate_config(config)
    original = {name: getattr(module, name) for name in ("Encoder", "SIGReg", "torch", "autocast", "GradScaler")}
    tensor_type = original["torch"].Tensor
    original_tensor_to = tensor_type.to

    def input_to_cpu(self, *args, **kwargs):
        if args:
            args = (_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = _device(kwargs["device"])
        return original_tensor_to(self, *args, **kwargs)

    class TorchDeviceProxy:
        def __getattr__(self, name):
            return getattr(original["torch"], name)

        def randn(self, *args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _device(kwargs["device"])
            return original["torch"].randn(*args, **kwargs)

    def to_cpu(self, *args, **kwargs):
        if args:
            args = (_device(args[0]), *args[1:])
        if "device" in kwargs:
            kwargs["device"] = _device(kwargs["device"])
        return super(type(self), self).to(*args, **kwargs)

    class CPUEncoder(original["Encoder"]):
        to = to_cpu

    class CPUSIGReg(original["SIGReg"]):
        to = to_cpu

    def scaler(*args, **kwargs):
        kwargs["enabled"] = False
        return original["GradScaler"](*args, **kwargs)

    metadata = dict(
        schema="cube_tiny_cpu_fp32_device_fixture_v1",
        role=config["numerical_fixture"]["role"],
        configured_batch_size=256,
        actual_fixed_batch_size=batch,
        device="cpu",
        precision="fp32",
        scaler_enabled=False,
        transformations=[
            "original Encoder/SIGReg .to(cuda) -> .to(cpu)",
            "original input Tensor.to(cuda) -> original Tensor.to(cpu)",
            "original SIGReg torch.randn(device=cuda) -> original torch.randn(device=cpu)",
            "old autocast -> explicit FP32 nullcontext; GradScaler explicitly disabled",
        ],
        fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_factory_source=dict(
            path=str(Path(inspect.getfile(original["Encoder"])).resolve()),
            sha256=hashlib.sha256(Path(inspect.getfile(original["Encoder"])).read_bytes()).hexdigest(),
        ),
        original_factory_init_reused=CPUEncoder.__init__ is original["Encoder"].__init__,
        original_factory_forward_reused=CPUEncoder.forward is original["Encoder"].forward,
    )
    if config.get("continuation_source") is not None:
        metadata["continuation_source"] = config["continuation_source"]
    module.Encoder, module.SIGReg = CPUEncoder, CPUSIGReg
    module.torch = TorchDeviceProxy()
    module.autocast = lambda *a, **k: nullcontext()
    module.GradScaler = scaler
    tensor_type.to = input_to_cpu
    try:
        yield metadata
    finally:
        tensor_type.to = original_tensor_to
        for name, value in original.items():
            setattr(module, name, value)
