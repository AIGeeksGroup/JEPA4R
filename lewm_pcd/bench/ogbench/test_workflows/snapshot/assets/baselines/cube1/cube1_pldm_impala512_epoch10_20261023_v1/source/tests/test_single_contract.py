import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import shared_data as d


def test_single_budget_and_episode_boundary():
    assert d.contract("pldm") == dict(
        frames=16,
        actions=15,
        per_episode=185,
        train_samples=1_850_000,
        validation_samples=185_000,
        train_batches=14_453,
        validation_batches=1_446,
        validation_tail=40,
        total_updates=144_530,
    )
    assert d.location(184, "pldm", 2) == (0, 184)
    assert d.location(185, "pldm", 2) == (1, 0)
    assert d.location(-1, "pldm", 2) == (1, 184)
    with pytest.raises(IndexError):
        d.location(370, "pldm", 2)


def test_reference_preprocessing_and_one_action_normalization():
    import torch
    from torchvision.transforms import v2

    class Reader:
        def __getitem__(self, index):
            return dict(
                pixels=torch.arange(16, dtype=torch.uint8)[:, None, None, None].expand(
                    16, 3, 4, 4
                ),
                action=torch.arange(80, dtype=torch.float32).reshape(16, 5),
            )

        def close(self):
            pass

    x = d.VisualDataset(
        "unused",
        "pldm",
        reader=Reader(),
        action_stats=dict(mean=[1] * 5, std=[2] * 5),
    )[184]
    assert set(x) == {"pixels", "actions", "sample_id"}
    assert x["sample_id"] == 184 and x["actions"].shape == (15, 5)
    assert torch.equal(
        x["actions"],
        (torch.arange(75).reshape(15, 5) - 1) / (2 + 1e-6),
    )
    torch.manual_seed(7)
    pixels = torch.randint(0, 256, (2, 16, 3, 224, 224), dtype=torch.uint8)
    expected = v2.Resize((65, 65), antialias=True)(pixels.float() / 255.0)
    out = d.PLDMPreprocess()(
        dict(
            pixels=pixels,
            actions=x["actions"].unsqueeze(0).expand(2, -1, -1).contiguous(),
        )
    )
    assert torch.equal(out["pixels"], expected)
    with pytest.raises(ValueError):
        d.PLDMPreprocess()(out)


def test_single_scheduler_budget():
    import torch
    from methods.pldm import method

    sys.path.insert(0, str(Path(__file__).parents[1] / "vendor/pldm/source"))
    opt = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.0007)
    scheduler = method.build_scheduler(
        opt,
        epochs=10,
        batch_steps=d.contract("pldm")["train_batches"],
    )
    assert scheduler.adjust_learning_rate(0) == 0
    assert scheduler.adjust_learning_rate(14_453) == pytest.approx(0.00035)
    assert scheduler.adjust_learning_rate(144_530) == pytest.approx(0.00000035)


def test_native_spt_batch_transfer_preprocesses_pixels_once_before_forward():
    import inspect
    import torch
    from types import SimpleNamespace

    from shared_data import PLDMPreprocess
    from train import native_module

    assert "gpu_preprocess" not in inspect.signature(native_module).parameters

    class Model(torch.nn.Module):
        def forward(self, pixels, actions):
            assert tuple(pixels.shape) == (2, 16, 3, 65, 65)
            assert pixels.dtype == torch.float32
            assert tuple(actions.shape) == (2, 15, 5)
            assert actions.dtype == torch.float32
            return {"prediction": pixels.mean() + actions.mean()}

    preprocess = PLDMPreprocess()
    module = native_module("pldm", SimpleNamespace(), Model())
    trainer = SimpleNamespace(
        training=False,
        validating=True,
        sanity_checking=True,
        testing=False,
        predicting=False,
        val_dataloaders=[
            SimpleNamespace(dataset=SimpleNamespace(gpu_transform=preprocess))
        ],
        datamodule=None,
    )
    module._trainer = trainer
    batch = {
        "pixels": torch.zeros(2, 16, 3, 224, 224, dtype=torch.uint8),
        "actions": torch.ones(2, 15, 5),
    }
    batch = module.on_after_batch_transfer(batch, dataloader_idx=0)
    assert batch["pixels"].dtype == torch.float32
    assert tuple(batch["pixels"].shape) == (2, 16, 3, 65, 65)
    assert preprocess.last_contract is not None
    result = module.forward(batch, "train")
    assert torch.isfinite(result["prediction"])


def test_sealed_single_rows200_first_last_boundary_and_validation_tail():
    import h5py
    import numpy as np
    import torch

    root = Path(os.environ["SINGLE_DATA_ROOT"])
    assert d.validate_manifest(root)["actual_exit_code"] == 0
    source_root = Path(__file__).parents[1]
    assert d.require_sources(source_root, root)["data_root"] == str(root.resolve())
    train = d.original_reader(root / "data_views" / d.TRAIN_VIEW, "pldm", root)
    val = d.original_reader(root / "data_views" / d.VAL_VIEW, "pldm", root)
    try:
        assert len(train) == 1_850_000 and len(val) == 185_000
        stats = d.action_statistics(train)
        assert stats["count"] == 2_000_000
        assert stats["source_split"] == "train"
        train_view = d.VisualDataset(
            root / "data_views" / d.TRAIN_VIEW,
            "pldm",
            reader=train,
            action_stats=stats,
        )
        with h5py.File(root / "data_views" / d.TRAIN_VIEW, "r", swmr=True) as raw:
            for index, episode, start in (
                (0, 0, 0),
                (184, 0, 184),
                (185, 1, 0),
                (1_849_999, 9999, 184),
            ):
                sample = train[index]
                begin = episode * 200 + start
                assert tuple(sample["pixels"].shape) == (16, 3, 224, 224)
                assert sample["pixels"].dtype == torch.uint8
                assert np.array_equal(
                    sample["pixels"][0].permute(1, 2, 0).numpy(),
                    raw["pixels"][begin],
                )
                assert np.array_equal(
                    sample["action"][:15].numpy(),
                    raw["action"][begin : begin + 15],
                )
        with h5py.File(root / "data_views" / d.VAL_VIEW, "r", swmr=True) as raw:
            for index, episode, start in (
                (0, 0, 0),
                (184, 0, 184),
                (185, 1, 0),
                (184_999, 999, 184),
            ):
                sample = val[index]
                begin = episode * 200 + start
                assert tuple(sample["pixels"].shape) == (16, 3, 224, 224)
                assert np.array_equal(
                    sample["pixels"][0].permute(1, 2, 0).numpy(),
                    raw["pixels"][begin],
                )
                assert np.array_equal(
                    sample["action"][:15].numpy(),
                    raw["action"][begin : begin + 15],
                )
        tail = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                val,
                list(range(184_960, 185_000)),
            ),
            batch_size=128,
            drop_last=False,
            num_workers=0,
        )
        batches = list(tail)
        assert len(batches) == 1 and batches[0]["pixels"].shape[0] == 40
    finally:
        train.close()
        val.close()
