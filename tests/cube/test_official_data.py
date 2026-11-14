"""Bounded downloader and raw-terminal checks; no external requests."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest


def load_api():
    path = Path(__file__).resolve().parents[2] / "jepa4r/data/cube/official_data.py"
    assert path.exists(), "official dataset preparation entry is missing"
    spec = importlib.util.spec_from_file_location("_cube_official_data_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data)), "ETag": "fixture-v1"}


def record(data=b"complete public dataset"):
    return dict(name="cube-single-play-v0.npz", url="https://example.test/data.npz", bytes=len(data))


def test_plan_is_read_only_and_has_train_and_val(tmp_path):
    api = load_api()
    output = tmp_path / "not-created"
    for cube in (1, 2, 3, 4):
        plan = api.build_plan(cube, output)
        assert plan["cube_count"] == cube
        assert set(plan["files"]) == {"train", "val"}
        assert all(item["bytes"] > 0 and item["url"].startswith("https://") for item in plan["files"].values())
    assert not output.exists()
    with pytest.raises(ValueError):
        api.build_plan(0, output)


def test_download_checks_bytes_before_publishing(tmp_path, monkeypatch):
    api = load_api()
    data = b"complete public dataset"
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: Response(data))
    result = api.download(record(data), tmp_path)
    assert Path(result["path"]).read_bytes() == data
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert not (tmp_path / (record()["name"] + ".part")).exists()


def test_existing_wrong_file_is_never_overwritten(tmp_path, monkeypatch):
    api = load_api()
    target = tmp_path / record()["name"]
    target.write_bytes(b"old user data")
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network must not run"))
    with pytest.raises(ValueError, match="size"):
        api.download(record(), tmp_path)
    assert target.read_bytes() == b"old user data"


def test_resume_requires_matching_partial_range(tmp_path, monkeypatch):
    api = load_api()
    data = b"complete public dataset"
    item = record(data)
    partial = tmp_path / (item["name"] + ".part")
    partial.write_bytes(data[:8])
    (tmp_path / (item["name"] + ".download.json")).write_text(json.dumps(item))

    def open_range(request, **kwargs):
        assert request.get_header("Range") == "bytes=8-"
        return Response(data[8:], 206, {"Content-Range": f"bytes 8-{len(data) - 1}/{len(data)}"})

    monkeypatch.setattr(api.urllib.request, "urlopen", open_range)
    assert Path(api.download(item, tmp_path)["path"]).read_bytes() == data


def test_ignored_range_preserves_partial_failure_evidence(tmp_path, monkeypatch):
    api = load_api()
    item = record()
    partial = tmp_path / (item["name"] + ".part")
    partial.write_bytes(b"complete")
    (tmp_path / (item["name"] + ".download.json")).write_text(json.dumps(item))
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: Response(b"complete public dataset"))
    with pytest.raises(ValueError, match="range"):
        api.download(item, tmp_path)
    assert partial.read_bytes() == b"complete"
    assert not (tmp_path / item["name"]).exists()


def test_bad_download_is_not_a_complete_dataset(tmp_path, monkeypatch):
    api = load_api()
    item = record(b"expected bytes")
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: Response(b"truncated"))
    with pytest.raises(ValueError):
        api.download(item, tmp_path)
    assert not (tmp_path / item["name"]).exists()
    assert (tmp_path / (item["name"] + ".part")).is_file()


def test_empty_interrupted_partial_can_resume(tmp_path, monkeypatch):
    api = load_api()
    data = b"complete public dataset"
    item = record(data)
    (tmp_path / (item["name"] + ".part")).write_bytes(b"")
    (tmp_path / (item["name"] + ".download.json")).write_text(json.dumps(item))
    monkeypatch.setattr(api.urllib.request, "urlopen", lambda *a, **k: Response(data))
    assert Path(api.download(item, tmp_path)["path"]).read_bytes() == data


def test_atomic_publish_never_replaces_another_writer(tmp_path):
    api = load_api()
    partial, target = tmp_path / "source.part", tmp_path / "source.npz"
    partial.write_bytes(b"new verified bytes")
    target.write_bytes(b"another writer won")
    with pytest.raises(FileExistsError):
        api.publish(partial, target)
    assert target.read_bytes() == b"another writer won"
    assert partial.read_bytes() == b"new verified bytes"


def write_fixture(path, *, bad_terminal=False, bad_qpos=False):
    qpos = np.zeros((1001, 16), np.float64)
    if bad_qpos:
        qpos[731, 4] = np.nan
    terminals = np.zeros(1001, np.float32)
    terminals[1000 if not bad_terminal else 999] = 1
    np.savez(
        path,
        observations=np.zeros((1001, 28), np.float32),
        actions=np.zeros((1001, 5), np.float32),
        terminals=terminals,
        qpos=qpos,
        qvel=np.zeros((1001, 14), np.float64),
    )


def test_audit_keeps_the_real_1001st_terminal(tmp_path):
    api = load_api()
    path = tmp_path / "source.npz"
    write_fixture(path)
    result = api.audit_npz(path, 1, expected_episodes=1)
    assert result["raw_rows"] == 1001
    assert result["episodes"] == 1
    assert result["transition_rows"] == 1000


@pytest.mark.parametrize("bad_terminal,bad_qpos", [(True, False), (False, True)])
def test_audit_rejects_the_exact_bad_terminal_or_nan_frame(tmp_path, bad_terminal, bad_qpos):
    api = load_api()
    path = tmp_path / "source.npz"
    write_fixture(path, bad_terminal=bad_terminal, bad_qpos=bad_qpos)
    with pytest.raises(ValueError):
        api.audit_npz(path, 1, expected_episodes=1)
