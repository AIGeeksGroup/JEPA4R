"""Narrow CPU checks; run with CUDA_VISIBLE_DEVICES='' and torch threads1."""
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

METHOD = Path(__file__).with_name("method.py")
CASE = Path(__file__).resolve().parents[2]
SOURCE = CASE / "vendor/pldm/source"


def load_method():
    spec = importlib.util.spec_from_file_location("cube_pldm_method", METHOD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_method_exists():
    assert METHOD.is_file(), "the new PLDM adapter has not been implemented"


def test_contract_rejects_action_chunk_and_state_fields():
    method = load_method()
    pixels = SimpleNamespace(shape=(2, 16, 3, 65, 65))
    method.check_batch(pixels, SimpleNamespace(shape=(2, 15, 5)))
    with pytest.raises(ValueError, match="15, 5"):
        method.check_batch(pixels, SimpleNamespace(shape=(2, 15, 25)))
    with pytest.raises(ValueError, match="16"):
        method.check_batch(SimpleNamespace(shape=(2, 4, 3, 65, 65)),
                           SimpleNamespace(shape=(2, 3, 5)))
    assert method.COUNTS == {"windows_per_episode": 585, "train_windows": 5850000,
        "validation_windows": 585000, "train_batches": 45703,
        "validation_batches": 4571, "validation_tail": 40, "total_updates": 457030}


def test_no_pretrained_or_import_time_torch():
    method = load_method()
    with pytest.raises(ValueError, match="pretrained"):
        method.build_model(SOURCE, pretrained=True)
    assert "import torch" not in inspect.getsource(method).split("def build_model")[0]


def test_cpu_original_math_optimizer_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    method = load_method()
    model = method.build_model(SOURCE)
    pixels = torch.rand(2, 16, 3, 65, 65)
    actions = torch.rand(2, 15, 5)
    result = model.forward_result(pixels, actions)
    assert result.backbone_output.encodings.shape == (16, 2, 512)
    assert model.core.backbone.stack_blocks[0].initial_conv.in_channels == 3
    assert model.core.backbone.mlp.in_features == 2592
    assert model.core.predictor.rnn.num_layers == 1
    assert model.core.predictor.final_ln is model.core.backbone.final_ln
    from pldm.models.jepa import JEPA
    from pldm.objectives.vicreg import VICRegObjective
    from pldm.objectives.idm import IDMObjective
    assert type(model.core).forward_posterior is JEPA.forward_posterior
    assert type(model.vicreg).__call__ is VICRegObjective.__call__
    assert type(model.idm).__call__ is IDMObjective.__call__
    direct_vic = VICRegObjective.__call__(model.vicreg, None, [result])
    direct_idm = IDMObjective.__call__(model.idm, None, [result])
    output = model(pixels, actions)
    assert torch.equal(output["loss"], direct_vic.total_loss + direct_idm.total_loss)
    assert tuple(inspect.signature(model.forward).parameters) == ("pixels", "actions")
    with pytest.raises(TypeError):
        model(pixels, actions, object_state=torch.ones(2, 3))
    optimizer = method.build_optimizer(model)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert {id(p) for p in model.parameters()} == optimized
    assert {id(p) for p in model.idm.parameters()} <= optimized
    before = {k: p.detach().clone() for k, p in model.idm.named_parameters()}
    output["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.idm.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in model.idm.named_parameters())
    checkpoint = tmp_path / "cpu_roundtrip.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored = method.build_model(SOURCE)
    restored_optimizer = method.build_optimizer(restored)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(saved["model"], strict=True)
    restored_optimizer.load_state_dict(saved["optimizer"])
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert any(k.startswith("idm.") for k in saved["model"])
    scheduler = method.build_scheduler(restored_optimizer, epochs=10, batch_steps=45703)
    assert scheduler.adjust_learning_rate(0) == 0
    assert scheduler.adjust_learning_rate(45703) == pytest.approx(0.00035)
    assert scheduler.adjust_learning_rate(457030) == pytest.approx(0.00000035)
