"""Read-only, all-frame RGB/XYZRGB review for single-front-camera pilot HDF5.

No simulator imports. Each episode gets an MP4, JSONL and CSV; summary.json is
written only after decoded MP4 frame counts equal the original ep_len values.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


CAMERA = "front_pixels"
CUBE_KEY = "privileged/block_0_pos"
TARGET_KEY = "privileged/target_block_pos"
HEADER_HEIGHT = 160


def _value(f, key, row):
    if key not in f:
        return None
    value = np.asarray(f[key][row])
    if value.size != 1:
        raise ValueError(f"{key} must contain one scalar per frame")
    value = value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _label(f, key, row, fallback):
    value = _value(f, key, row)
    if fallback is not None and value is not None and str(value) != str(fallback):
        raise ValueError(f"Explicit {key}={fallback!r} conflicts with HDF5 {value!r} at row {row}")
    return fallback if value is None else value


def _position(f, key, row):
    if key not in f:
        return None
    value = np.asarray(f[key][row], dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{key} must have shape (frames, 3)")
    return value if np.isfinite(value).all() else None


def frame_metrics(f, *, episode, step, row, ep_offset, ep_len, task_id=None,
                  strategy=None, cube_key=CUBE_KEY, target_key=TARGET_KEY, near_radius=0.04):
    xyz = np.asarray(f["pcd_xyz"][row])
    colors = np.asarray(f["pcd_rgb"][row])
    finite = np.isfinite(xyz).all(axis=1)
    points = xyz[finite]
    cube, target = _position(f, cube_key, row), _position(f, target_key, row)
    return {
        "episode": int(episode), "step": int(step), "global_row": int(row),
        "ep_offset": int(ep_offset), "ep_len": int(ep_len), "camera": CAMERA,
        "task_id": _label(f, "task_id", row, task_id),
        "strategy": _label(f, "strategy", row, strategy),
        "point_count": int(len(xyz)), "finite_xyz_count": int(finite.sum()),
        "nonfinite_xyz_count": int((~finite).sum()),
        "nonfinite_rgb_count": int((~np.isfinite(colors).all(axis=1)).sum()),
        "xyz_min": points.min(axis=0).tolist() if len(points) else None,
        "xyz_max": points.max(axis=0).tolist() if len(points) else None,
        "cube_position": cube.tolist() if cube is not None else None,
        "target_position": target.tolist() if target is not None else None,
        "near_cube_count": int((np.linalg.norm(points - cube, axis=1) < near_radius).sum()) if cube is not None else None,
        "near_cube_radius": float(near_radius),
        "cube_target_distance": float(np.linalg.norm(cube - target)) if cube is not None and target is not None else None,
    }


def _fmt(value):
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(f"{v:.4f}" for v in value) + "]"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _projection(points, colors, bounds, axes, size, cube, target):
    panel = Image.new("RGB", (size, size), (25, 29, 32))
    draw = ImageDraw.Draw(panel)
    a, b = axes
    low, high = np.asarray(bounds)
    margin, top, bottom = 18, 25, size - 22
    x0, x1 = margin, size - margin
    draw.rectangle((x0, top, x1, bottom), outline=(80, 88, 94))

    def project(values):
        u = x0 + (values[..., a] - low[a]) / (high[a] - low[a]) * (x1 - x0)
        v = bottom - (values[..., b] - low[b]) / (high[b] - low[b]) * (bottom - top)
        return np.stack([u, v], axis=-1)

    # Draw all points; the hidden coordinate determines deterministic occlusion.
    hidden = ({0, 1, 2} - {a, b}).pop()
    for index in np.argsort(points[:, hidden], kind="stable"):
        u, v = project(points[index])
        if x0 <= u <= x1 and top <= v <= bottom:
            draw.rectangle((round(u), round(v), round(u) + 1, round(v) + 1), fill=tuple(colors[index]))
    for pos, color in ((cube, (255, 255, 255)), (target, (0, 255, 255))):
        if pos is not None:
            u, v = project(np.asarray(pos))
            if x0 <= u <= x1 and top <= v <= bottom:
                draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=color)
    draw.text((4, 4), "XYZRGB " + "XYZ"[a] + "/" + "XYZ"[b], fill="white")
    draw.text((4, size - 17), f"{'XYZ'[a]} {low[a]:.3f}..{high[a]:.3f}", fill="white")
    return panel


def render_frame(f, row, record, bounds, *, panel_size=256):
    xyz, colors = np.asarray(f["pcd_xyz"][row]), np.asarray(f["pcd_rgb"][row])
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(colors).all(axis=1)
    xyz, colors = xyz[valid], colors[valid]
    cube, target = record["cube_position"], record["target_position"]
    rgb = Image.fromarray(np.asarray(f[f"pixels/{CAMERA}"][row])).resize((panel_size, panel_size), Image.Resampling.NEAREST)
    ImageDraw.Draw(rgb).text((4, 4), f"RGB {CAMERA}", fill="white", stroke_width=1, stroke_fill="black")
    panels = [rgb] + [_projection(xyz, colors, bounds, axes, panel_size, cube, target) for axes in ((0, 1), (0, 2), (1, 2))]
    frame = Image.new("RGB", (4 * panel_size, HEADER_HEIGHT + panel_size), (9, 12, 15))
    for index, panel in enumerate(panels):
        frame.paste(panel, (index * panel_size, HEADER_HEIGHT))
    lines = [
        f"task={_fmt(record['task_id'])} strategy={_fmt(record['strategy'])} episode={record['episode']} step={record['step']}/{record['ep_len'] - 1} row={row}",
        f"N={record['point_count']} finiteXYZ={record['finite_xyz_count']} nonfiniteXYZ={record['nonfinite_xyz_count']} nonfiniteRGB={record['nonfinite_rgb_count']}",
        f"xyz min={_fmt(record['xyz_min'])} max={_fmt(record['xyz_max'])}",
        f"cube={_fmt(cube)} target={_fmt(target)}",
        f"cube-target distance={_fmt(record['cube_target_distance'])} near-cube N={_fmt(record['near_cube_count'])} radius={record['near_cube_radius']}",
        "Fixed episode axes; cube=white ring; target=cyan ring; missing fields=null",
    ]
    draw = ImageDraw.Draw(frame)
    y = 4
    for line in lines:
        for wrapped in textwrap.wrap(line, width=max(1, (frame.width - 8) // 6)):
            draw.text((4, y), wrapped, fill="white")
            y += 14
    return np.asarray(frame)


def _validate(f, episodes, task_id, strategy, cube_key, target_key):
    for key in ("ep_offset", "ep_len", "pcd_xyz", "pcd_rgb", f"pixels/{CAMERA}"):
        if key not in f:
            raise ValueError(f"Missing required HDF5 field: {key}")
    if sorted(f["pixels"].keys()) != [CAMERA]:
        raise ValueError(f"Expected a single {CAMERA} camera, found {list(f['pixels'].keys())}")
    xyz, colors, pixels = f["pcd_xyz"], f["pcd_rgb"], f[f"pixels/{CAMERA}"]
    if xyz.ndim != 3 or xyz.shape[2] != 3 or xyz.shape[1] == 0:
        raise ValueError("pcd_xyz must have shape (frames, points, 3) with nonempty points")
    if colors.shape != xyz.shape:
        raise ValueError("pcd_rgb and pcd_xyz shapes must agree")
    if colors.dtype != np.uint8:
        raise ValueError("pcd_rgb must be uint8 RGB with values in [0, 255]")
    if pixels.ndim != 4 or pixels.shape[0] != xyz.shape[0] or pixels.shape[3] != 3 or pixels.dtype != np.uint8:
        raise ValueError("front_pixels must be uint8 (frames, height, width, 3) aligned with pcd_xyz")
    lengths, offsets = f["ep_len"][:], f["ep_offset"][:]
    if lengths.ndim != 1 or offsets.ndim != 1 or len(lengths) != len(offsets) or not np.issubdtype(lengths.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("ep_offset and ep_len must be matching integer vectors")
    expected = np.cumsum(np.r_[0, lengths[:-1]], dtype=np.int64)
    if not len(lengths) or np.any(lengths <= 0) or not np.array_equal(offsets, expected) or int(lengths.sum()) != len(xyz):
        raise ValueError("ep_offset/ep_len must cover all frames exactly without gaps or overlaps")
    selected = list(range(len(lengths))) if episodes is None else list(episodes)
    if not selected or len(selected) != len(set(selected)) or any(ep < 0 or ep >= len(lengths) for ep in selected):
        raise ValueError("episodes must be nonempty, unique, valid zero-based episode indices")
    for key in ("task_id", "strategy", cube_key, target_key):
        if key in f and (f[key].ndim == 0 or len(f[key]) != len(xyz)):
            raise ValueError(f"{key} must be aligned with all frames")
    for ep in selected:
        for row in range(int(offsets[ep]), int(offsets[ep] + lengths[ep])):
            _label(f, "task_id", row, task_id)
            _label(f, "strategy", row, strategy)
            _position(f, cube_key, row)
            _position(f, target_key, row)
    return selected, offsets, lengths


def _episode_bounds(f, offset, length, cube_key, target_key):
    lows, highs = [], []
    for row in range(offset, offset + length):
        xyz = np.asarray(f["pcd_xyz"][row])
        finite = xyz[np.isfinite(xyz).all(axis=1)]
        if len(finite):
            lows.append(finite.min(axis=0))
            highs.append(finite.max(axis=0))
        for key in (cube_key, target_key):
            pos = _position(f, key, row)
            if pos is not None:
                lows.append(pos)
                highs.append(pos)
    if not lows:
        return [[-1.0] * 3, [1.0] * 3]
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    padding = np.maximum((high - low) * 0.05, 0.01)
    return [(low - padding).tolist(), (high + padding).tolist()]


def review_pilot(source, output, *, episodes=None, fps=20, panel_size=256,
                 task_id=None, strategy=None, cube_key=CUBE_KEY, target_key=TARGET_KEY, near_radius=0.04):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {output}")
    if fps <= 0 or panel_size < 128 or panel_size % 16 or near_radius <= 0:
        raise ValueError("fps/radius must be positive and panel_size must be >=128 and divisible by 16")
    with h5py.File(source, "r") as f:
        selected, offsets, lengths = _validate(f, episodes, task_id, strategy, cube_key, target_key)
        try:
            from imageio.plugins.ffmpeg import get_exe
            get_exe()
        except ImportError as exc:
            raise RuntimeError("MP4 review requires the existing imageio_ffmpeg dependency; none was installed") from exc
        output.mkdir(parents=True, exist_ok=False)
        source_hash = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_hash.update(chunk)
        summary = {"source": str(source), "source_sha256": source_hash.hexdigest(), "camera": CAMERA,
                   "cube_key": cube_key, "target_key": target_key, "near_radius": near_radius,
                   "fps": fps, "panel_size": panel_size, "total_frames": 0, "episodes": []}
        for ep in selected:
            offset, length = int(offsets[ep]), int(lengths[ep])
            bounds = _episode_bounds(f, offset, length, cube_key, target_key)
            prefix = output / f"episode_{ep:04d}"
            video_path = prefix.with_suffix(".mp4")
            with prefix.with_suffix(".jsonl").open("x", encoding="utf-8") as jsonl, prefix.with_suffix(".csv").open("x", newline="", encoding="utf-8") as table, imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
                csv_writer = None
                for step, row in enumerate(range(offset, offset + length)):
                    record = frame_metrics(f, episode=ep, step=step, row=row, ep_offset=offset,
                                           ep_len=length, task_id=task_id, strategy=strategy,
                                           cube_key=cube_key, target_key=target_key, near_radius=near_radius)
                    jsonl.write(json.dumps(record, allow_nan=False) + "\n")
                    if csv_writer is None:
                        csv_writer = csv.DictWriter(table, fieldnames=list(record))
                        csv_writer.writeheader()
                    csv_writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, list) else value for key, value in record.items()})
                    writer.append_data(render_frame(f, row, record, bounds, panel_size=panel_size))
            with imageio.get_reader(video_path) as reader:
                actual = reader.count_frames()
            if actual != length:
                raise RuntimeError(f"{video_path.name}: decoded {actual} frames, expected ep_len={length}")
            summary["episodes"].append({"episode": int(ep), "ep_offset": offset, "ep_len": length,
                                        "verified_video_frames": actual, "bounds": bounds,
                                        "video": video_path.name, "jsonl": prefix.name + ".jsonl", "csv": prefix.name + ".csv"})
            summary["total_frames"] += length
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", help="Zero-based episodes; default: every episode")
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--task-id", type=int, help="Explicit pilot task label when HDF5 task_id is absent")
    parser.add_argument("--strategy", help="Explicit pilot strategy label when HDF5 strategy is absent")
    parser.add_argument("--cube-key", default=CUBE_KEY)
    parser.add_argument("--target-key", default=TARGET_KEY)
    parser.add_argument("--near-radius", type=float, default=0.04)
    args = parser.parse_args()
    summary = review_pilot(**vars(args))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
