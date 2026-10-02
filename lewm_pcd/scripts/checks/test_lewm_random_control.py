import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from hydra.utils import call, get_method
from torch import nn

from bench.cortexbench import lewm_cortex


MODEL_CONFIGS = Path(lewm_cortex.__file__).parent / "conf" / "model"


def _small_projector(cfg):
    return nn.Sequential(
        nn.Linear(cfg["input_dim"], cfg["hidden_dim"]),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(cfg["hidden_dim"], cfg["output_dim"]),
    )


def _assert_same_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.fixture(autouse=True)
def isolated_rng():
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture(params=[True, False], ids=["xyz_rgb", "xyz"])
def checkpoint(request, tmp_path, monkeypatch):
    kwargs = {"center": [0.075, -0.02, 0.225], "scale": 4.0, "use_rgb": request.param}
    cfg = {
        "encoder": {
            "_target_": "encoders.DexArtPointNet",
            "in_dim": 6 if request.param else 3,
            "mlp_spec": [8, 16],
            "out_dim": 192,
            "chunk_points": 300,
        },
        "projector": {
            "_target_": "module.MLP",
            "input_dim": 192,
            "hidden_dim": 16,
            "output_dim": 192,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"data": {"preprocessor": kwargs, "dataset": {"name": "test-pen"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(lewm_cortex, "instantiate", _small_projector)
    return tmp_path, cfg, kwargs


@pytest.fixture
def pretrained_checkpoint(checkpoint):
    ckdir, cfg, _ = checkpoint
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        encoder = lewm_cortex._dexart_pointnet_class()(
            **{key: value for key, value in cfg["encoder"].items() if key != "_target_"}
        )
        model = lewm_cortex.LeWMEmbedding(encoder, _small_projector(cfg["projector"]))
        torch.save(model.state_dict(), ckdir / "weights_epoch_7.pt")
    return checkpoint, model.eval()


def test_random_pcd_default_seed_matches_explicit_seed_across_caller_rng(checkpoint):
    ckdir, _, kwargs = checkpoint
    torch.manual_seed(100)
    first, _, _, _ = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, **kwargs)
    torch.manual_seed(200)
    repeat, _, _, _ = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, seed=0, **kwargs)

    _assert_same_state(first.state_dict(), repeat.state_dict())
    inputs = torch.linspace(-1, 1, 2 * 1200 * (6 if kwargs["use_rgb"] else 3)).reshape(2, 1200, -1)
    with torch.no_grad():
        torch.testing.assert_close(first(inputs), repeat(inputs), rtol=0, atol=0)


def test_random_pcd_seed_changes_encoder_and_projector(checkpoint):
    ckdir, _, kwargs = checkpoint
    first, _, _, _ = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, seed=0, **kwargs)
    other, _, _, metadata = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, seed=1, **kwargs)

    for component in ("encoder", "projector"):
        first_state = getattr(first, component).state_dict()
        other_state = getattr(other, component).state_dict()
        assert any(not torch.equal(first_state[key], other_state[key]) for key in first_state)
    assert metadata["initialization_seed"] == 1


def test_random_pcd_skips_weights_and_returns_native_finite_eval_embedding(checkpoint, monkeypatch):
    ckdir, _, kwargs = checkpoint

    def fail_weight_read(*args, **kwargs):
        pytest.fail("random initialization must not read checkpoint weights")

    monkeypatch.setattr(torch, "load", fail_weight_read)
    model, dim, transform, metadata = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, **kwargs)

    assert model.encoder.__class__.__name__ == "DexArtPointNet"
    assert not any(module.training for module in model.modules())
    assert dim == 192
    assert metadata == {
        "algo": "random",
        "input": "pcd",
        "model": "DexArt PointNet (LeWM 3D encoder + projector)",
        "data": [],
        "checkpoint": None,
        "initialization_seed": 0,
    }
    pcd = torch.cat([torch.tensor(kwargs["center"]).expand(1200, 3), torch.full((1200, 3), 255)], dim=-1)
    inputs = transform(pcd).unsqueeze(0).expand(2, -1, -1)
    assert inputs.shape == (2, 1200, 6 if kwargs["use_rgb"] else 3)
    assert torch.equal(inputs[..., :3], torch.zeros_like(inputs[..., :3]))
    if kwargs["use_rgb"]:
        assert torch.equal(inputs[..., 3:], torch.ones_like(inputs[..., 3:]))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        output = model(inputs)
        assert output.shape == (2, dim)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, model(inputs), rtol=0, atol=0)
    _assert_same_state(model.state_dict(), before)


@pytest.mark.parametrize("seed", [0, 19])
def test_random_pcd_consumes_same_rng_as_pretrained(pretrained_checkpoint, seed):
    (ckdir, _, kwargs), reference = pretrained_checkpoint
    torch.manual_seed(777)
    pretrained, _, _, _ = lewm_cortex.load_lewm_pcd(ckdir, epoch=7, **kwargs)
    pretrained_rng = torch.get_rng_state().clone()
    pretrained_draws = torch.rand(16)

    torch.manual_seed(777)
    random_model, _, _, _ = lewm_cortex.load_lewm_pcd(ckdir, random_init=True, seed=seed, **kwargs)

    assert torch.equal(torch.get_rng_state(), pretrained_rng)
    assert torch.equal(torch.rand(16), pretrained_draws)
    _assert_same_state(pretrained.state_dict(), reference.state_dict())
    for component in ("encoder", "projector"):
        pretrained_state = getattr(pretrained, component).state_dict()
        random_state = getattr(random_model, component).state_dict()
        assert pretrained_state.keys() == random_state.keys()
        assert all(pretrained_state[key].shape == random_state[key].shape for key in random_state)
        assert any(not torch.equal(pretrained_state[key], random_state[key]) for key in random_state)


@pytest.mark.parametrize("options", [{}, {"random_init": False, "seed": 19}])
def test_pretrained_pcd_matches_original_construction_and_metadata(pretrained_checkpoint, options):
    (ckdir, cfg, kwargs), saved = pretrained_checkpoint
    torch.manual_seed(555)
    encoder = lewm_cortex._dexart_pointnet_class()(
        **{key: value for key, value in cfg["encoder"].items() if key != "_target_"}
    )
    original = lewm_cortex.LeWMEmbedding(encoder, _small_projector(cfg["projector"]))
    original.load_state_dict(saved.state_dict(), strict=True)
    original.eval()
    with torch.no_grad():
        original_dim = original(torch.zeros(2, 1200, cfg["encoder"]["in_dim"])).shape[-1]
    original_rng = torch.get_rng_state().clone()

    torch.manual_seed(555)
    model, dim, transform, metadata = lewm_cortex.load_lewm_pcd(ckdir, epoch=7, **kwargs, **options)

    assert torch.equal(torch.get_rng_state(), original_rng)
    assert dim == original_dim == 192
    _assert_same_state(model.state_dict(), saved.state_dict())
    assert not any(module.training for module in model.modules())
    assert metadata == {
        "algo": "lewm",
        "input": "pcd",
        "model": "DexArt PointNet (LeWM 3D encoder + projector)",
        "data": ["test-pen"],
        "checkpoint": str(ckdir / "weights_epoch_7.pt"),
    }
    pcd = torch.linspace(0, 255, 1200 * 6).reshape(1200, 6)
    expected_transform = lewm_cortex.PcdTransform(**kwargs)
    torch.testing.assert_close(transform(pcd), expected_transform(pcd), rtol=0, atol=0)
    with torch.no_grad():
        inputs = transform(pcd).unsqueeze(0)
        torch.testing.assert_close(model(inputs), original(inputs), rtol=0, atol=0)


def test_random_pcd_preserves_preprocessor_validation(checkpoint):
    ckdir, _, kwargs = checkpoint
    with pytest.raises(AssertionError, match="preprocessor mismatch"):
        lewm_cortex.load_lewm_pcd(ckdir, random_init=True, **{**kwargs, "scale": 2.0})


@pytest.mark.parametrize("modality", ["2d", "pcd"])
def test_pen_random_configs_match_pretrained_architecture_and_resolve_loader(modality, monkeypatch):
    base = yaml.safe_load((MODEL_CONFIGS / f"lewm_adroit_pen_{modality}.yaml").read_text())
    random_cfg = yaml.safe_load((MODEL_CONFIGS / f"lewm_adroit_pen_{modality}_random.yaml").read_text())
    assert random_cfg == {**base, "random_init": True, "seed": 0}
    monkeypatch.setitem(sys.modules, "lewm_cortex", lewm_cortex)
    expected = lewm_cortex.load_lewm if modality == "2d" else lewm_cortex.load_lewm_pcd
    assert get_method(random_cfg["_target_"]) is expected


def test_pen_random_pcd_config_calls_loader_through_hydra(checkpoint, monkeypatch):
    ckdir, _, kwargs = checkpoint
    cfg = yaml.safe_load((MODEL_CONFIGS / "lewm_adroit_pen_pcd_random.yaml").read_text())
    cfg.update(checkpoint_dir=str(ckdir), **kwargs)
    monkeypatch.setitem(sys.modules, "lewm_cortex", lewm_cortex)
    model, dim, _, metadata = call(cfg)
    assert dim == 192 and not model.training
    assert metadata["algo"] == "random" and metadata["initialization_seed"] == 0
