"""CPU-only fixed-view visualization of exact XYZ+RGB encoder input points.

This module draws the provided points; it never loads a simulator, mesh, image
background, or learned model. Pixel markers represent the original samples and
do not densify the point cloud. One input array frame becomes one video frame.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


def encode_pointcloud_video(path, pointclouds, fps=20, *, size=640,
                            azimuth_degrees=135.0, elevation_degrees=25.0):
    """Encode [T,N,6] XYZ+RGB at a fixed orthographic view using CPU only.

    Returns JSON-serializable provenance, verification, and preview paths. RGB
    values in [0,1] or [0,255] are supported. All input points are preserved;
    depth sorting only determines the order of rasterized point markers.
    """
    output = Path(path)
    if output.exists():
        raise FileExistsError(output)
    original = np.asarray(pointclouds)
    if original.ndim != 3 or original.shape[-1] != 6 or min(original.shape[:2]) < 1:
        raise ValueError(f"Expected nonempty [T,N,6] XYZ+RGB, got {original.shape}")
    if not np.isfinite(original).all() or fps <= 0 or size < 64 or size % 2:
        raise ValueError("Require finite points, positive fps, and an even image size >= 64")
    points = np.ascontiguousarray(original, dtype=np.float32)
    xyz = points[..., :3]
    colors = points[..., 3:]
    if colors.min() < 0 or colors.max() > 255:
        raise ValueError("RGB values must be within [0,1] or [0,255]")
    rgb_scale = 255.0 if colors.max() <= 1.0 else 1.0
    colors_u8 = np.rint(colors * rgb_scale).astype(np.uint8)
    azimuth, elevation = np.deg2rad([azimuth_degrees, elevation_degrees])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.array([-np.sin(elevation) * np.cos(azimuth),
                   -np.sin(elevation) * np.sin(azimuth), np.cos(elevation)])
    toward_camera = np.array([np.cos(elevation) * np.cos(azimuth),
                              np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    rotation = np.stack([right, up, toward_camera], axis=1)
    projected = xyz @ rotation
    lower = projected[..., :2].min(axis=(0, 1))
    upper = projected[..., :2].max(axis=(0, 1))
    center = (lower + upper) / 2
    span = max(float(np.max(upper - lower)), 1e-6)
    pixels_per_unit = 0.84 * size / span
    coords = (projected[..., :2] - center) * pixels_per_unit
    coords[..., 0] += size / 2
    coords[..., 1] = size / 2 - coords[..., 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = {}
    preview_indices = {0, len(points) // 2, len(points) - 1}
    with imageio.get_writer(str(output), format="FFMPEG", mode="I", fps=fps,
                            codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                            output_params=["-crf", "16", "-threads", "1", "-movflags", "+faststart"]) as writer:
        for frame_index, (frame_coords, frame_colors, frame_projected) in enumerate(
                zip(coords, colors_u8, projected)):
            frame = Image.new("RGB", (size, size), (17, 21, 29))
            draw = ImageDraw.Draw(frame)
            order = np.argsort(frame_projected[:, 2], kind="stable")
            for point_index in order:
                x, y = frame_coords[point_index]
                draw.ellipse((float(x - 1.5), float(y - 1.5), float(x + 1.5), float(y + 1.5)),
                             fill=tuple(int(c) for c in frame_colors[point_index]))
            writer.append_data(np.asarray(frame))
            if frame_index in preview_indices:
                preview = output.with_name(f"{output.stem}_frame{frame_index:04d}.png")
                frame.save(preview)
                previews[str(frame_index)] = str(preview.resolve())
    decoded_count, duration = imageio_ffmpeg.count_frames_and_secs(str(output))
    with imageio.get_reader(str(output), format="FFMPEG") as reader:
        metadata = reader.get_meta_data()
        shapes = {reader.get_data(i).shape for i in preview_indices}
    expected_shape = (size, size, 3)
    if decoded_count != len(points) or abs(duration - len(points) / fps) >= 1 / fps:
        raise RuntimeError("Encoded video length does not match the input trajectory")
    if shapes != {expected_shape}:
        raise RuntimeError(f"Unexpected decoded image shapes: {shapes}")
    return {
        "video": str(output.resolve()),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "input_shape": list(points.shape), "original_input_dtype": str(original.dtype),
        "visualized_input_dtype": "float32",
        "input_float32_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "points_per_frame": int(points.shape[1]), "fps": float(fps),
        "rgb_input_range": [float(colors.min()), float(colors.max())],
        "rgb_to_uint8_scale": rgb_scale,
        "visualization": "Fixed-view orthographic rasterization of exact XYZ+RGB points; no mesh, camera RGB background, additional points, interpolation, or frame omission.",
        "view": {"azimuth_degrees": float(azimuth_degrees), "elevation_degrees": float(elevation_degrees),
                 "projection": "orthographic", "range_policy": "all points across the complete episode, fixed for every frame",
                 "xyz_min": xyz.min(axis=(0, 1)).tolist(), "xyz_max": xyz.max(axis=(0, 1)).tolist(),
                 "projected_xy_min": lower.tolist(), "projected_xy_max": upper.tolist(),
                 "point_marker_diameter_pixels": 3.0, "background_rgb": [17, 21, 29]},
        "verification": {"decoded_frames": decoded_count, "duration_seconds": duration,
                         "decoded_shape_hwc": list(expected_shape), "codec": metadata.get("codec"),
                         "pixel_format": metadata.get("pix_fmt"), "fps": metadata.get("fps")},
        "preview_frames": previews,
    }
