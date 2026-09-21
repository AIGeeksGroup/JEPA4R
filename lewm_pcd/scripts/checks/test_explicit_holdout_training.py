from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
import pytest
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import RandomSampler

import train


class Dataset:
    def __init__(self, name):
        self.name = name
        self.transform = None


class GroupedDataset(TorchDataset):
    def __init__(self, episodes=8, clips_per_episode=20):
        self.episode_ids = [
            episode
            for episode in range(episodes)
            for _ in range(clips_per_episode)
        ]

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        return index


def test_explicit_holdout_is_loaded_as_a_separate_dataset(monkeypatch):
    primary = Dataset("train.h5")
    holdout = Dataset("holdout.h5")
    calls = []

    def fake_load(name, **kwargs):
        calls.append((name, kwargs))
        return holdout

    monkeypatch.setattr(train.swm.data, "load_dataset", fake_load)
    cfg = OmegaConf.create(
        {"data": {"validation_name": "task/holdout.h5"}, "train_split": 0.9}
    )
    train_set, val_set = train.resolve_train_val_datasets(
        primary,
        cfg,
        dataset_kwargs={"num_steps": 4, "frameskip": 5},
        cache_dir="/cache",
        generator=torch.Generator().manual_seed(1),
    )
    assert train_set is primary
    assert val_set is holdout
    assert calls == [
        (
            "task/holdout.h5",
            {
                "transform": None,
                "cache_dir": "/cache",
                "num_steps": 4,
                "frameskip": 5,
            },
        )
    ]


def test_legacy_config_keeps_random_split(monkeypatch):
    primary = Dataset("legacy.h5")
    sentinel = (object(), object())
    calls = []

    def fake_split(dataset, lengths, generator):
        calls.append((dataset, lengths, generator))
        return sentinel

    monkeypatch.setattr(train.spt.data, "random_split", fake_split)
    cfg = OmegaConf.create({"data": {}, "train_split": 0.8})
    generator = torch.Generator().manual_seed(2)
    result = train.resolve_train_val_datasets(
        primary,
        cfg,
        dataset_kwargs={},
        cache_dir=None,
        generator=generator,
    )
    assert result is sentinel
    assert len(calls) == 1
    assert calls[0][0] is primary
    assert calls[0][1] == pytest.approx([0.8, 0.2])
    assert calls[0][2] is generator


def test_train_and_validation_loaders_use_independent_seeded_full_batches():
    cfg = OmegaConf.create(
        {
            "seed": 3072,
            "loader": {
                "batch_size": 32,
                "num_workers": 0,
                "persistent_workers": False,
            },
        }
    )
    dataset = GroupedDataset()

    train_loader, val_loader = train.make_train_val_loaders(dataset, dataset, cfg)
    _, val_loader_again = train.make_train_val_loaders(dataset, dataset, cfg)

    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, RandomSampler)
    assert train_loader.drop_last is True
    assert val_loader.drop_last is True
    assert train_loader.generator is not val_loader.generator
    assert train_loader.generator.initial_seed() == 3072
    assert val_loader.generator.initial_seed() == 3072

    first_batch = next(iter(val_loader)).tolist()
    repeated_batch = next(iter(val_loader_again)).tolist()
    assert first_batch == repeated_batch
    assert len({dataset.episode_ids[index] for index in first_batch}) > 1
