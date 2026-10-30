"""Synthetic, simulator-free checks for the full-frame pilot reviewer."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pytest


MODULE = Path(__file__).with_name("review_pcd_pilot.py")


def load_reviewer():
    assert MODULE.is_file(), "The offline pilot reviewer is not implemented yet"
    spec = importlib.util.spec_from_file_location("review_pcd_pilot", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pilot(path, *, optional=True, invalid_offset=False, extra_camera=False):
    cube = np.array([0.4, 0.0, 0.05], dtype=np.float32)
    xyz = np.stack([cube, cube + [0.02, 0, 0], cube + [0.05, 0, 0], cube + [0.07, 0, 0]])
    xyz = np.repeat(xyz[None], 5, axis=0).astype(np.float32)
    xyz[4, 3] = np.nan
    with h5py.File(path, "w") as f:
        f["ep_offset"] = [0, 4] if invalid_offset else [0, 3]
        f["ep_len"] = [3, 2]
        f["pcd_xyz"] = xyz
        f["pcd_rgb"] = np.repeat(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]], dtype=np.uint8), 5, axis=0)
        f["pixels/front_pixels"] = np.repeat(np.arange(5, dtype=np.uint8)[:, None, None, None], 16 * 16 * 3, axis=1).reshape(5, 16, 16, 3)
        if extra_camera:
            f["pixels/side_pixels"] = np.zeros((5, 16, 16, 3), dtype=np.uint8)
        if optional:
            f["task_id"] = [1, 1, 1, 2, 2]
            f.create_dataset("strategy", data=["low_noise"] * 3 + ["random"] * 2, dtype=h5py.string_dtype())
            f["privileged/block_0_pos"] = np.repeat(cube[None], 5, axis=0)
            f["privileged/target_block_pos"] = np.repeat((cube + [0.05, 0, 0])[None], 5, axis=0)


def test_exports_every_frame_and_preserves_input(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "review"
    summary = reviewer.review_pilot(source, output, episodes=[0, 1], fps=5, panel_size=128)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert summary["camera"] == "front_pixels"
    assert summary["total_frames"] == 5
    assert [entry["ep_offset"] for entry in summary["episodes"]] == [0, 3]
    assert [entry["ep_len"] for entry in summary["episodes"]] == [3, 2]
    assert [entry["verified_video_frames"] for entry in summary["episodes"]] == [3, 2]
    records = []
    for ep in (0, 1):
        records.extend(json.loads(line) for line in (output / f"episode_{ep:04d}.jsonl").read_text().splitlines())
        with imageio.get_reader(output / f"episode_{ep:04d}.mp4") as video:
            assert video.count_frames() == (3 if ep == 0 else 2)
            assert video.get_data(0).shape[:2] == (288, 512)
        with (output / f"episode_{ep:04d}.csv").open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == (3 if ep == 0 else 2)
    assert [record["global_row"] for record in records] == list(range(5))
    assert [record["step"] for record in records] == [0, 1, 2, 0, 1]
    assert records[0]["task_id"] == 1
    assert records[3]["strategy"] == "random"
    assert records[0]["point_count"] == 4
    assert records[4]["finite_xyz_count"] == 3
    assert records[4]["nonfinite_xyz_count"] == 1
    assert records[0]["near_cube_count"] == 2
    assert records[0]["cube_target_distance"] == pytest.approx(0.05)
    assert records[0]["xyz_min"] == pytest.approx([0.4, 0, 0.05])
    assert json.loads((output / "summary.json").read_text()) == summary


def test_missing_metadata_is_null_and_single_episode_selection_is_exact(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source, optional=False)
    output = tmp_path / "review"
    summary = reviewer.review_pilot(source, output, episodes=[1], fps=5, panel_size=128)
    records = [json.loads(line) for line in (output / "episode_0001.jsonl").read_text().splitlines()]
    assert summary["total_frames"] == 2
    assert [record["global_row"] for record in records] == [3, 4]
    for record in records:
        for key in ("task_id", "strategy", "cube_target_distance", "near_cube_count"):
            assert record[key] is None
    assert not (output / "episode_0000.mp4").exists()


def test_explicit_pilot_labels_are_used_when_hdf5_labels_are_absent(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source, optional=False)
    output = tmp_path / "review"
    reviewer.review_pilot(source, output, episodes=[0], fps=5, panel_size=128,
                          task_id=3, strategy="expert_then_random")
    record = json.loads((output / "episode_0000.jsonl").read_text().splitlines()[0])
    assert record["task_id"] == 3
    assert record["strategy"] == "expert_then_random"


def test_explicit_labels_must_not_conflict_with_hdf5(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source)
    with pytest.raises(ValueError, match="task_id"):
        reviewer.review_pilot(source, tmp_path / "review", episodes=[0], task_id=4, panel_size=128)


@pytest.mark.parametrize("invalid_offset,extra_camera,message", [
    (True, False, "ep_offset"),
    (False, True, "single"),
])
def test_rejects_invalid_episode_boundaries_and_extra_cameras(tmp_path, invalid_offset, extra_camera, message):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source, invalid_offset=invalid_offset, extra_camera=extra_camera)
    with pytest.raises(ValueError, match=message):
        reviewer.review_pilot(source, tmp_path / "review", episodes=[0], panel_size=128)


def test_existing_output_is_never_overwritten(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source)
    output = tmp_path / "review"
    output.mkdir()
    sentinel = output / "already_here.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        reviewer.review_pilot(source, output, episodes=[0], panel_size=128)
    assert sentinel.read_text() == "preserve"


def test_frame_metrics_and_three_world_coordinate_panels(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source)
    with h5py.File(source, "r") as f:
        record = reviewer.frame_metrics(f, episode=1, step=1, row=4, ep_offset=3, ep_len=2)
        assert record["finite_xyz_count"] == 3
        assert record["nonfinite_xyz_count"] == 1
        assert record["xyz_min"] == pytest.approx([0.4, 0, 0.05])
        assert record["xyz_max"] == pytest.approx([0.45, 0, 0.05])
        frame = reviewer.render_frame(f, 4, record, [[0.35, -0.05, 0], [0.5, 0.05, 0.1]], panel_size=128)
    assert frame.dtype == np.uint8
    assert frame.shape == (288, 512, 3)
    # Each orthographic panel must contain actual point colors, not a blank frame.
    for start in (128, 256, 384):
        panel = frame[160:, start:start + 128]
        assert np.any(np.all(panel == [0, 0, 255], axis=-1))


@pytest.mark.parametrize("dtype,values", [
    (np.float32, [0.2, 0.5, 0.8]),
    (np.int16, [300, -10, 60]),
    (np.float32, [np.nan, 60, 80]),
    (np.uint16, [255, 60, 80]),
])
def test_rejects_non_uint8_point_colors_before_creating_output(tmp_path, dtype, values):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_pilot(source)
    with h5py.File(source, "r+") as f:
        del f["pcd_rgb"]
        f["pcd_rgb"] = np.broadcast_to(np.array(values, dtype=dtype), (5, 4, 3))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "review"
    with pytest.raises(ValueError, match="pcd_rgb.*uint8"):
        reviewer.review_pilot(source, output, episodes=[0], panel_size=128)
    assert not output.exists()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


ORDER_COLORS = np.array([[192, 40, 40], [40, 192, 40], [40, 40, 192],
                         [192, 192, 40], [192, 40, 192]], dtype=np.uint8)


def write_order_pilot(path):
    write_pilot(path, optional=False)
    grid_x, grid_y = np.meshgrid(np.arange(10) * 0.003, np.arange(10) * 0.003)
    patch = np.stack([grid_x.ravel(), grid_y.ravel(), grid_y.ravel()], axis=1)
    xyz = np.stack([patch + [0.4 + 0.08 * row, -0.2 + 0.04 * row, 0.05 + 0.03 * row]
                    for row in range(5)]).astype(np.float32)
    with h5py.File(path, "r+") as f:
        del f["pcd_xyz"], f["pcd_rgb"], f["pixels/front_pixels"]
        f["pcd_xyz"] = xyz
        f["pcd_rgb"] = np.repeat(ORDER_COLORS[:, None], 100, axis=1)
        f["pixels/front_pixels"] = np.broadcast_to((20 + np.arange(5) * 45).astype(np.uint8)[:, None, None, None], (5, 16, 16, 3))


def assert_frame_content(frame, row, bounds, source):
    assert frame.shape == (416, 1024, 3)
    # The central RGB crop avoids labels and has a distinct luminance per row.
    assert np.median(frame[224:352, 64:192]) == pytest.approx(20 + 45 * row, abs=8)
    low, high = np.asarray(bounds)
    center = source["pcd_xyz"][row].mean(axis=0)
    for index, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), start=1):
        panel = frame[160:, index * 256:(index + 1) * 256].astype(float)
        mask = np.linalg.norm(panel - ORDER_COLORS[row], axis=-1) < 70
        v, u = np.nonzero(mask)
        assert len(u) >= 12, f"Missing row {row} point colors in projection {a}/{b}"
        expected_u = 18 + (center[a] - low[a]) / (high[a] - low[a]) * 220
        expected_v = 234 - (center[b] - low[b]) / (high[b] - low[b]) * 209
        # MP4 encoding shifts sparse colored points by a few pixels; the
        # direct-render tests above retain the tighter geometric check.
        assert np.mean(u) == pytest.approx(expected_u, abs=8)
        assert np.mean(v) == pytest.approx(expected_v, abs=8)


def test_rendered_rgb_and_xyzrgb_follow_source_rows(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_order_pilot(source)
    with h5py.File(source, "r") as f:
        for ep, (offset, length) in enumerate(((0, 3), (3, 2))):
            bounds = reviewer._episode_bounds(f, offset, length, reviewer.CUBE_KEY, reviewer.TARGET_KEY)
            for step, row in enumerate(range(offset, offset + length)):
                record = reviewer.frame_metrics(f, episode=ep, step=step, row=row, ep_offset=offset, ep_len=length)
                assert_frame_content(reviewer.render_frame(f, row, record, bounds), row, bounds, f)


def test_decoded_video_rgb_and_xyzrgb_preserve_every_frame_order(tmp_path):
    reviewer = load_reviewer()
    source = tmp_path / "pilot.h5"
    write_order_pilot(source)
    output = tmp_path / "review"
    summary = reviewer.review_pilot(source, output, fps=5)
    with h5py.File(source, "r") as f:
        for entry in summary["episodes"]:
            with imageio.get_reader(output / entry["video"]) as reader:
                assert reader.count_frames() == entry["ep_len"]
                for step in range(entry["ep_len"]):
                    frame = reader.get_data(step)
                    row = entry["ep_offset"] + step
                    assert frame.shape == (416, 1024, 3)
                    assert np.median(frame[224:352, 64:192]) == pytest.approx(20 + 45 * row, abs=8)
                    for index in range(1, 4):
                        panel = frame[160:, index * 256:(index + 1) * 256].astype(float)
                        mask = np.linalg.norm(panel - ORDER_COLORS[row], axis=-1) < 70
                        assert np.count_nonzero(mask) >= 12
