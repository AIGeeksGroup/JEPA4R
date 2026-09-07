"""Read-only bounded pointcloud audit of the numeric audit's fixed Lance version.

Exactly 32 uniformly sampled complete episodes; only those episodes' pointclouds
are read. CPU only. A missing red cube candidate is not a task-failure label.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import lancedb

HERE = Path(__file__).resolve().parent
SEED = 20260907
SAMPLE_EPISODES = 32
H = 200
DT = .05
BASE_OFFSET = np.asarray([.615, 0., 0.], dtype=np.float32)
NEAR_CUBE_RADIUS_M = .045
THRESHOLDS = {
    "loose": {"red_min": .20, "red_over_other_ratio": 1.5, "red_minus_other_min": .08},
    "strict": {"red_min": .35, "red_over_other_ratio": 2.0, "red_minus_other_min": .15},
}


def quantiles(x):
    x = np.asarray(x).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return {k: float(v) for k, v in zip(
        ["min", "p05", "p25", "median", "p75", "p95", "max"],
        np.quantile(x, [0, .05, .25, .5, .75, .95, 1]))}


def longest_run(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    if not len(starts):
        return {"frames": 0, "seconds": 0., "start_step": None, "end_step_inclusive": None}
    i = int(np.argmax(ends - starts))
    n = int(ends[i] - starts[i])
    return {"frames": n, "seconds": n * DT, "start_step": int(starts[i]), "end_step_inclusive": int(ends[i] - 1)}


def detection_stats(counts):
    counts = np.asarray(counts)
    return {
        "frames": int(counts.size), "point_count_quantiles": quantiles(counts),
        "zero_point_frames": int((counts == 0).sum()),
        "zero_point_fraction": float(np.mean(counts == 0)),
        "under_five_point_frames": int((counts < 5).sum()),
        "under_five_point_fraction": float(np.mean(counts < 5)),
    }


def numeric(arrow, name):
    values = arrow[name].combine_chunks()
    return values.values.to_numpy(zero_copy_only=False).reshape(len(values), values.type.list_size)


def main():
    source = json.loads((HERE / "summary.json").read_text())["snapshot"]
    ep_ids = np.load(HERE / "numeric_episode_snapshot.npz")["episode_ids"]
    sampled = np.sort(np.random.default_rng(SEED).choice(ep_ids, size=SAMPLE_EPISODES, replace=False))
    path = Path(source["dataset"])
    table = lancedb.connect(str(path.parent)).open_table(path.stem)
    table.checkout(source["lance_version"])
    assert table.version == source["lance_version"]
    per_episode = []
    all_counts = {name: [] for name in THRESHOLDS}
    near_counts = {name: [] for name in THRESHOLDS}
    far_counts = {name: [] for name in THRESHOLDS}
    distances = {name: [] for name in THRESHOLDS}
    centroids, mins, maxs, centroid_deltas = [], [], [], []
    heights_all, tcp_all = [], []
    nonfinite = 0
    rgb_min = np.full(3, np.inf)
    rgb_max = np.full(3, -np.inf)
    rows_read = 0
    stored_dtype = None
    for episode_id in sampled:
        arrow = (table.search().where(f"episode_idx = {int(episode_id)}")
                 .select(["episode_idx", "step_idx", "pointcloud", "cube_pos", "object_height_above_table", "tcp_object_distance"])
                 .limit(H + 1).to_arrow().sort_by([("step_idx", "ascending")]))
        assert len(arrow) == H
        assert arrow["step_idx"].to_pylist() == list(range(H))
        stored = numeric(arrow, "pointcloud")
        stored_dtype = str(stored.dtype)
        assert stored.shape == (H, 1200 * 6)
        pc = stored.astype(np.float32).reshape(H, 1200, 6)
        xyz, rgb = pc[:, :, :3], pc[:, :, 3:]
        nonfinite += int(np.count_nonzero(~np.isfinite(pc)))
        assert np.isfinite(pc).all()
        cube_base = numeric(arrow, "cube_pos") + BASE_OFFSET
        cube_distance = np.linalg.norm(xyz - cube_base[:, None, :], axis=-1)
        close = cube_distance <= NEAR_CUBE_RADIUS_M
        heights = numeric(arrow, "object_height_above_table").reshape(-1)
        tcp = numeric(arrow, "tcp_object_distance").reshape(-1)
        heights_all.append(heights)
        tcp_all.append(tcp)
        centroid = xyz.mean(axis=1)
        min_xyz, max_xyz = xyz.min(axis=1), xyz.max(axis=1)
        centroids.append(centroid)
        mins.append(min_xyz)
        maxs.append(max_xyz)
        centroid_deltas.append(np.linalg.norm(np.diff(centroid, axis=0), axis=-1))
        rgb_min = np.minimum(rgb_min, rgb.min(axis=(0, 1)))
        rgb_max = np.maximum(rgb_max, rgb.max(axis=(0, 1)))
        episode = {
            "episode_idx": int(episode_id), "frames": H,
            "centroid_xyz_first_middle_last": centroid[[0, 100, 199]].tolist(),
            "xyz_min_first_middle_last": min_xyz[[0, 100, 199]].tolist(),
            "xyz_max_first_middle_last": max_xyz[[0, 100, 199]].tolist(),
            "centroid_step_displacement_m": quantiles(np.linalg.norm(np.diff(centroid, axis=0), axis=-1)),
            "thresholds": {},
        }
        r = rgb[:, :, 0]
        other = np.maximum(rgb[:, :, 1], rgb[:, :, 2])
        for name, params in THRESHOLDS.items():
            red = ((r > params["red_min"]) & (r > params["red_over_other_ratio"] * other)
                   & ((r - other) > params["red_minus_other_min"]))
            counts = red.sum(axis=1)
            confirmed = (red & close).sum(axis=1)
            far = (red & ~close).sum(axis=1)
            all_counts[name].append(counts)
            near_counts[name].append(confirmed)
            far_counts[name].append(far)
            distances[name].append(cube_distance[red])
            episode["thresholds"][name] = {
                **detection_stats(confirmed), "color_only": detection_stats(counts),
                "longest_zero_detection_run": longest_run(confirmed == 0),
                "longest_under_five_run": longest_run(confirmed < 5),
                "cube_candidate_points_by_step": confirmed.tolist(),
                "red_points_by_step": counts.tolist(),
                "time_thirds": [detection_stats(part) for part in np.array_split(confirmed, 3)],
            }
        per_episode.append(episode)
        rows_read += H
        print(f"audited {len(per_episode)}/{SAMPLE_EPISODES} episodes, {rows_read} pointcloud frames", flush=True)
    centroid = np.concatenate(centroids)
    lower, upper = np.concatenate(mins), np.concatenate(maxs)
    heights, tcp = np.stack(heights_all), np.stack(tcp_all)
    result = {
        "snapshot": {"source_dataset": str(path), "lance_version": source["lance_version"],
                     "numeric_snapshot_utc": source["snapshot_utc"], "audit_utc": datetime.now(timezone.utc).isoformat(),
                     "sample_seed": SEED, "sampling": "uniform random without replacement over complete snapshot episode IDs",
                     "population_episodes": len(ep_ids), "episode_ids": sampled.tolist(),
                     "episodes_analyzed": SAMPLE_EPISODES, "frames_analyzed": rows_read,
                     "columns_read": ["episode_idx", "step_idx", "pointcloud", "cube_pos", "object_height_above_table", "tcp_object_distance"]},
        "integrity": {"points_per_frame": 1200, "channels": ["x", "y", "z", "r", "g", "b"],
                      "stored_dtype": stored_dtype, "inspection_dtype": "float32 (same boundary cast as encoder)",
                      "nonfinite_values": nonfinite, "rgb_min": rgb_min.tolist(), "rgb_max": rgb_max.tolist()},
        "method": {"color_thresholds": THRESHOLDS, "geometric_confirmation_radius_m": NEAR_CUBE_RADIUS_M,
                   "cube_world_to_pointcloud_base_translation_xyz": BASE_OFFSET.tolist(),
                   "meaning": "cube candidates satisfy a red-color threshold AND lie within 4.5cm of recorded cube center in the pointcloud coordinate system",
                   "limitations": ["No red cube candidate is a sampled visual detection proxy; not proof that the physical cube is invisible or that the task failed.",
                                   "Both thresholds can miss very dark cube surfaces; geometric matching reduces false positives, not false negatives.",
                                   "32 episodes are a limited uniform sample; temporal frames are correlated, so frame counts are not independent trials.",
                                   "Grasp contact was not recorded; elevation and TCP distance do not establish contact-based native grasp."]},
        "pointcloud_geometry": {
            "centroid_xyz_min": centroid.min(axis=0).tolist(), "centroid_xyz_max": centroid.max(axis=0).tolist(),
            "global_xyz_min": lower.min(axis=0).tolist(), "global_xyz_max": upper.max(axis=0).tolist(),
            "frame_centroid_step_displacement_m": quantiles(np.concatenate(centroid_deltas)),
            "frame_bounds_extent_m": {axis: quantiles((upper-lower)[:, i]) for i, axis in enumerate("xyz")},
        },
        "thresholds": {},
    }
    for name in THRESHOLDS:
        counts = np.stack(near_counts[name])
        color_counts = np.stack(all_counts[name])
        total_red = int(color_counts.sum())
        far = int(np.stack(far_counts[name]).sum())
        runs = [(entry["episode_idx"], entry["thresholds"][name]["longest_zero_detection_run"]) for entry in per_episode]
        longest = max(runs, key=lambda row: row[1]["frames"])
        bands = {}
        for band, mask in {"height_under_5cm": heights < .05, "height_5_to_15cm": (heights >= .05) & (heights < .15),
                           "height_at_least_15cm": heights >= .15, "tcp_within_6cm": tcp <= .06, "tcp_beyond_6cm": tcp > .06}.items():
            bands[band] = detection_stats(counts[mask]) if mask.any() else None
        result["thresholds"][name] = {
            **detection_stats(counts), "color_only": detection_stats(color_counts),
            "red_points_outside_geometric_radius": far,
            "fraction_red_points_outside_radius": far / total_red if total_red else None,
            "red_point_to_cube_center_distance_m": quantiles(np.concatenate(distances[name])),
            "episodes_with_any_zero_detection": int((counts == 0).any(axis=1).sum()),
            "episodes_with_any_under_five": int((counts < 5).any(axis=1).sum()),
            "longest_zero_detection_run": {"episode_idx": longest[0], **longest[1]},
            "episode_longest_zero_run_frames": quantiles([entry[1]["frames"] for entry in runs]),
            "time_thirds": [{"steps": [int(ids[0]), int(ids[-1])], **detection_stats(counts[:, ids])}
                            for ids in np.array_split(np.arange(H), 3)],
            "height_and_tcp_bands": bands,
        }
    (HERE / "visual-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "visual_episode_metrics.json").write_text(json.dumps(per_episode, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in metrics.items()
                            if key in ("zero_point_fraction", "under_five_point_fraction", "longest_zero_detection_run", "fraction_red_points_outside_radius")}
                      for name, metrics in result["thresholds"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
