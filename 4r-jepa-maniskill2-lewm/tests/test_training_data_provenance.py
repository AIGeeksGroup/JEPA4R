"""Only explicitly marked non-training data is refused; external data stays valid."""
import json
from pathlib import Path

import h5py
import pytest

from ms2_lewm.official import load_source

ROOT = Path(__file__).resolve().parents[1]
driver = load_source("_production_dataset_guard", ROOT / "scripts/train_wm.py")


@pytest.mark.parametrize("suffix", [".lance", ".h5"])
@pytest.mark.parametrize("marker", ["SYNTHETIC_UNIT_TEST_ONLY", "BENCHMARK_ONLY_NOT_TRAINING_DATA"])
def test_explicit_sidecar_markers_are_refused(tmp_path, suffix, marker):
    path = tmp_path / ("dataset" + suffix)
    path.with_suffix(suffix + ".json").write_text(json.dumps({marker: True}))
    with pytest.raises(ValueError, match="cannot be used for production training"):
        driver.validate_production_dataset(path)


def test_old_hdf5_attribute_still_refused(tmp_path):
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["SYNTHETIC_UNIT_TEST_ONLY"] = True
    with pytest.raises(ValueError, match="cannot be used for production training"):
        driver.validate_production_dataset(path)


def test_unmarked_external_dataset_and_explicit_false_marker_remain_allowed(tmp_path):
    path = tmp_path / "external.lance"
    path.mkdir()
    driver.validate_production_dataset(path)
    path.with_suffix(".lance.json").write_text(json.dumps({"BENCHMARK_ONLY_NOT_TRAINING_DATA": False}))
    driver.validate_production_dataset(path)


def test_guard_runs_before_cuda_admission(monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "benchmark.lance"
    path.with_suffix(".lance.json").write_text(json.dumps({"BENCHMARK_ONLY_NOT_TRAINING_DATA": True}))
    monkeypatch.setattr(driver.subprocess, "check_output", lambda *a, **kw: pytest.fail("CUDA admission was reached"))
    with pytest.raises(ValueError, match="cannot be used for production training"):
        driver.run(SimpleNamespace(smoke=False, dataset=str(path)))
