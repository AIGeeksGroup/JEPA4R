#!/usr/bin/env python3
"""Export exact recorded Lance pointcloud inputs (or optional camera RGB)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import imageio_ffmpeg
import lancedb
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ms2_lewm.pointcloud_video import encode_pointcloud_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the existing .lance table")
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--mode", choices=("pointcloud", "rgb"), default="pointcloud")
    args = parser.parse_args()
    source = args.dataset.resolve(strict=True)
    if source.suffix != ".lance" or args.episode < 0 or args.fps <= 0:
        parser.error("Require a .lance table, non-negative episode, and positive fps")
    table = lancedb.connect(str(source.parent)).open_table(source.stem)
    data_column = "pointcloud" if args.mode == "pointcloud" else "pixels"
    columns = ["episode_idx", "step_idx", data_column, "seed"]
    rows = table.search().where(f"episode_idx = {args.episode}").limit(100000).select(columns).to_arrow()
    rows = rows.sort_by([("step_idx", "ascending")]).to_pylist()
    if not rows:
        raise ValueError(f"Episode {args.episode} has no frames")
    steps = [int(row["step_idx"]) for row in rows]
    if steps != list(range(len(rows))):
        raise ValueError("Episode frames must have consecutive unique step indices starting at 0")
    seeds = {int(np.asarray(row["seed"]).item()) for row in rows}
    if len(seeds) != 1:
        raise ValueError(f"Unexpected multiple episode seeds: {seeds}")
    seed = seeds.pop()
    report_path = source.parent / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    dataset_manifest_path = source.with_suffix(source.suffix + ".json")
    dataset_manifest = json.loads(dataset_manifest_path.read_text()) if dataset_manifest_path.exists() else {}
    provenance = {**report, **dataset_manifest}
    benchmark_only = bool(provenance.get("BENCHMARK_ONLY_NOT_TRAINING_DATA", False)
                          or provenance.get("benchmark_only", False))
    records = [row for row in report.get("episode_records", []) if row["seed"] == seed]
    if len(records) > 1:
        raise ValueError("Cannot uniquely match episode to its native success report")
    record = records[0] if records else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    common_manifest = {
        "source_dataset": str(source),
        "source_report": str(report_path) if report_path.exists() else None,
        "source_manifest": str(dataset_manifest_path) if dataset_manifest_path.exists() else None,
        "episode_idx": args.episode, "seed": seed, "steps": len(rows),
        "native_success": record.get("success"),
        "native_success_source": "episode_records entry matched by seed" if record else "not inferred from appearance",
        "policy": provenance.get("policy", "CubeMarkovOracle + original ExpertPolicy noise through ManiSkill state/action adapter"),
        "oracle_mode": provenance.get("oracle_mode"),
        "noise_config": provenance.get("noise_config"),
        "benchmark_only": benchmark_only,
        "official_expert_dataset": provenance.get("official_expert_dataset"),
        "note": ("Real recorded benchmark trajectory; not a training dataset."
                 if benchmark_only else "Real trajectory from the collection dataset; not selected by success or failure."),
        "sampling_stride_control_steps": 1,
        "source_mode": args.mode,
    }
    if args.mode == "pointcloud":
        # Keep the actual storage precision (fp16 in current compact data),
        # then apply only the encoder boundary's float32 conversion.
        stored_dtype = np.dtype(table.schema.field("pointcloud").type.value_type.to_pandas_dtype())
        stored_points = np.asarray([row["pointcloud"] for row in rows], dtype=stored_dtype)
        stored_points = stored_points.reshape(len(rows), 1200, 6)
        encoder_points = stored_points.astype(np.float32)
        raw_path = args.output.with_suffix(".input.npz")
        if raw_path.exists():
            raise FileExistsError(raw_path)
        stored_key = "stored_pointcloud_fp16" if stored_dtype == np.float16 else "stored_pointcloud"
        np.savez_compressed(raw_path, pointcloud=encoder_points,
                            step_idx=np.asarray(steps, dtype=np.int32),
                            **{stored_key: stored_points})
        manifest = dict(common_manifest)
        manifest.update(encode_pointcloud_video(args.output, encoder_points, fps=args.fps))
        manifest.update({
            "input_archive": str(raw_path.resolve()),
            "input_archive_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "source_pointcloud_dtype": str(stored_dtype),
            "stored_pointcloud_sha256": hashlib.sha256(stored_points.tobytes()).hexdigest(),
            "encoder_input_contract": f"Lance pointcloud {stored_dtype} values cast directly to float32, reshape [T,1200,6]; XYZ+RGB only, identical to world-model input before PointNet.",
        })
        args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    previews = {}
    source_hash = hashlib.sha256()
    frame_shape = None
    with imageio.get_writer(str(args.output), format="FFMPEG", mode="I", fps=args.fps,
                            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            output_params=["-crf", "18", "-threads", "1", "-movflags", "+faststart"]) as writer:
        for index, row in enumerate(rows):
            source_hash.update(row["pixels"])
            frame = np.asarray(Image.open(io.BytesIO(row["pixels"])).convert("RGB"))
            if frame_shape is None:
                frame_shape = frame.shape
            if frame.shape != frame_shape:
                raise ValueError("Recorded frame dimensions changed during the episode")
            writer.append_data(frame)
            if index in {0, len(rows) // 2, len(rows) - 1}:
                preview = args.output.with_name(f"{args.output.stem}_frame{index:04d}.png")
                Image.fromarray(frame).save(preview)
                previews[str(index)] = str(preview.resolve())
    decoded_count, duration = imageio_ffmpeg.count_frames_and_secs(str(args.output))
    with imageio.get_reader(str(args.output), format="FFMPEG") as reader:
        metadata = reader.get_meta_data()
        decoded_shapes = {reader.get_data(i).shape for i in (0, len(rows) // 2, len(rows) - 1)}
    assert decoded_count == len(rows), (decoded_count, len(rows))
    assert abs(duration - len(rows) / args.fps) < 1 / args.fps
    assert decoded_shapes == {frame_shape}, (decoded_shapes, frame_shape)
    manifest = {
        **common_manifest,
        "source_jpeg_sequence_sha256": source_hash.hexdigest(),
        "original_frame_shape_hwc": list(frame_shape), "fps": args.fps,
        "sampling_stride_control_steps": 1,
        "video": str(args.output.resolve()), "video_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "encoding": "H.264 / yuv420p / CRF 18; original resolution and temporal order; no interpolation",
        "verification": {"decoded_frames": decoded_count, "duration_seconds": duration,
                         "decoded_shape_hwc": list(frame_shape), "codec": metadata.get("codec"),
                         "pixel_format": metadata.get("pix_fmt"), "fps": metadata.get("fps")},
        "preview_frames": previews,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
