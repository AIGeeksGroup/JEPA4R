"""Generate a small, self-contained Adroit RGB/PCD geometry preview.

This is a review artifact only. It does not write a training dataset and does not
claim that the Pen controller or the formal sampling protocol has passed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
CB_ROOT = REPO / "bench" / "cortexbench"
PINNED = Path(os.environ.get(
    "CORTEXBENCH_PINNED_SRC",
    "/root/autodl-tmp/workspace/users/lewm/cortexbench-official-pinned-src",
))
MJ_ENVS = next(PINNED.glob("mj_envs-*"), PINNED / "mj_envs")
MJRL = next(PINNED.glob("mjrl-*"), PINNED / "mjrl")
for path in (REPO, CB_ROOT, MJ_ENVS, MJRL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", "/root/.mujoco/mujoco200")
os.environ.setdefault("MUJOCO_PY_MJKEY_PATH", "/root/.mujoco/mjkey.txt")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
_mujoco_bin = Path(os.environ["MUJOCO_PY_MUJOCO_PATH"]) / "bin"
os.environ["LD_LIBRARY_PATH"] = f"{_mujoco_bin}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")

from bench.cortexbench.adroit.pcd import AdroitPCDConfig, depth_to_world, sample_adroit_pcd  # noqa: E402


def _env_class(task: str):
    if task == "pen-v0":
        from mj_envs.envs.hand_manipulation_suite.pen_v0 import PenEnvV0

        return PenEnvV0
    if task == "relocate-v0":
        from mj_envs.envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0

        return RelocateEnvV0
    raise ValueError(f"unsupported preview task: {task}")


def _descendants(model, root_body: int) -> set[int]:
    parent = np.asarray(model.body_parentid, dtype=np.int64)
    out = {int(root_body)}
    changed = True
    while changed:
        changed = False
        for body_id, parent_id in enumerate(parent):
            if int(parent_id) in out and body_id not in out:
                out.add(body_id)
                changed = True
    return out


def _geom_masks(model) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    robot_root = int(model.body_name2id("forearm"))
    object_body = int(model.body_name2id("Object"))
    robot_bodies = _descendants(model, robot_root)
    object_bodies = _descendants(model, object_body)
    target_bodies = _descendants(model, int(model.body_name2id("target"))) if model.nbody else set()
    geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
    foreground = np.isin(geom_body, list(robot_bodies | object_bodies | target_bodies))
    object_geom = np.isin(geom_body, list(object_bodies))
    labels = {
        "robot_bodies": [model.body_id2name(i) for i in sorted(robot_bodies)],
        "object_bodies": [model.body_id2name(i) for i in sorted(object_bodies)],
        "target_bodies": [model.body_id2name(i) for i in sorted(target_bodies)],
    }
    return foreground, object_geom, labels


def _depth_buffer_to_metric(sim, depth: np.ndarray) -> np.ndarray:
    extent = float(sim.model.stat.extent)
    near = float(sim.model.vis.map.znear) * extent
    far = float(sim.model.vis.map.zfar) * extent
    return near / (1.0 - depth.astype(np.float64) * (1.0 - near / far))


def _png_data_url(rgb: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _capture(sim, config: AdroitPCDConfig, foreground: np.ndarray, object_geom: np.ndarray, rng) -> dict:
    size = config.image_size
    cam = config.camera
    # Segmentation IDs are only reliable when multisampling is disabled before
    # the first offscreen context is created.
    sim.model.vis.quality.offsamples = 0
    rgb, depth = sim.render(width=size, height=size, depth=True, camera_name=cam, device_id=0)
    seg = sim.render(width=size, height=size, segmentation=True, camera_name=cam, device_id=0)
    rgb = np.asarray(rgb)[::-1]
    depth = np.asarray(depth)[::-1]
    seg = np.asarray(seg)[::-1]
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise RuntimeError(f"unexpected segmentation shape: {seg.shape}")
    geom = np.where(seg[..., 0].astype(np.int64) == 5, seg[..., 1].astype(np.int64), -1)
    depth_metric = _depth_buffer_to_metric(sim, depth)
    cam_id = int(sim.model.camera_name2id(cam))
    xyz = depth_to_world(
        depth_metric,
        sim.data.cam_xpos[cam_id],
        np.asarray(sim.data.cam_xmat[cam_id]).reshape(3, 3),
        float(sim.model.cam_fovy[cam_id]),
    )
    flat_xyz = xyz.reshape(-1, 3)
    flat_rgb = rgb.reshape(-1, 3)
    flat_geom = geom.reshape(-1)
    valid = (flat_geom >= 0) & (flat_geom < len(foreground))
    valid &= foreground[np.clip(flat_geom, 0, len(foreground) - 1)]
    finite = np.isfinite(flat_xyz).all(axis=1)
    valid &= finite
    out_xyz, out_rgb, stats = sample_adroit_pcd(
        flat_xyz,
        flat_rgb,
        flat_geom,
        foreground,
        object_geom,
        rng,
        config,
    )
    stats.update({
        "rgb_height": int(rgb.shape[0]),
        "rgb_width": int(rgb.shape[1]),
        "finite_depth_fraction": float(np.isfinite(depth_metric).mean()),
        "foreground_pixel_fraction": float(valid.mean()),
        "visible_geom_count": int(np.unique(flat_geom[valid]).size),
        "xyz_min": np.min(out_xyz, axis=0).round(5).tolist(),
        "xyz_max": np.max(out_xyz, axis=0).round(5).tolist(),
    })
    return {
        "rgb": _png_data_url(rgb),
        "points": np.concatenate([out_xyz, out_rgb.astype(np.float32)], axis=1).round(5).tolist(),
        "stats": stats,
    }


def _bounds(frames: list[dict]) -> dict[str, list[float]]:
    points = np.asarray([point for frame in frames for point in frame["points"]], dtype=np.float32)
    xyz = points[:, :3]
    mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
    padded = np.maximum((maxs - mins) * 0.08, 0.01)
    mins -= padded
    maxs += padded
    return {"x": [float(mins[0]), float(maxs[0])], "y": [float(mins[1]), float(maxs[1])], "z": [float(mins[2]), float(maxs[2])]}


def _write_html(path: Path, task: str, seed: int, frames: list[dict], labels: dict[str, list[str]]) -> None:
    payload = {"task": task, "seed": seed, "status": "GEOMETRY_PREVIEW_ONLY", "bounds": _bounds(frames), "frames": frames, "masks": labels}
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adroit PCD preview</title>
<style>
:root{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;color:#e9edf2;background:#101419}
body{margin:0;background:#101419}.wrap{max-width:1500px;margin:0 auto;padding:24px}.head{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;border-bottom:1px solid #303841;padding-bottom:18px}.kicker{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:#91a3b5}.title{font-size:28px;font-weight:700;margin:6px 0}.sub{color:#aab5c0;font-size:14px}.badge{border:1px solid #d4a72c;color:#f1cc65;padding:8px 10px;font-size:12px;white-space:nowrap}.controls{display:flex;gap:10px;align-items:center;margin:18px 0}.controls button{background:#202832;color:#eef3f7;border:1px solid #3b4651;padding:8px 12px;cursor:pointer}.controls button:hover{background:#2b3743}.controls input{flex:1}.frame{color:#b9c4cf;font-variant-numeric:tabular-nums;min-width:100px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{background:#171d23;border:1px solid #2d3741;padding:12px}.panel h2{font-size:13px;font-weight:600;margin:0 0 10px;color:#dfe7ee}.panel canvas{display:block;width:100%;height:auto;background:#0b0f13;aspect-ratio:1/1}.meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:14px 0}.stat{background:#171d23;border-left:3px solid #5f7891;padding:9px 10px}.stat b{display:block;color:#edf2f5;font-size:13px}.stat span{display:block;color:#9caab6;font-size:12px;margin-top:3px}.note{margin-top:18px;color:#96a4af;font-size:13px;line-height:1.5}.warn{color:#f1cc65}.axes{color:#748392;font-size:11px;margin-top:6px}@media(max-width:900px){.grid{grid-template-columns:1fr}.meta{grid-template-columns:repeat(2,minmax(0,1fr))}.head{display:block}.badge{display:inline-block;margin-top:14px}}
</style></head><body><main class="wrap"><header class="head"><div><div class="kicker">CortexBench / Adroit</div><div class="title">Pen point-cloud geometry preview</div><div class="sub" id="subtitle"></div></div><div class="badge">GEOMETRY PREVIEW ONLY</div></header>
<div class="controls"><button id="prev">Previous</button><input id="slider" type="range" min="0" max="0" value="0"><button id="next">Next</button><span class="frame" id="frameLabel"></span></div><section class="meta" id="meta"></section><section class="grid"><div class="panel"><h2>Official RGB camera: vil_camera</h2><canvas id="rgb"></canvas></div><div class="panel"><h2>Point cloud: XY / top view</h2><canvas id="xy"></canvas><div class="axes">horizontal X, vertical Y</div></div><div class="panel"><h2>Point cloud: XZ / front view</h2><canvas id="xz"></canvas><div class="axes">horizontal X, vertical Z</div></div><div class="panel"><h2>Point cloud: YZ / side view</h2><canvas id="yz"></canvas><div class="axes">horizontal Y, vertical Z</div></div></section><p class="note"><span class="warn">Review boundary:</span> this page checks the official camera, depth back-projection, foreground mask, crop, voxel sampling, RGB/PCD frame alignment, and finite values. It is not a full three-part controller pilot, paired HDF5 smoke, or full-collection approval.</p></main>
<script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent), frames=data.frames; let i=0;
const $=id=>document.getElementById(id), slider=$('slider'); slider.max=frames.length-1;
function fit(v,a,b){return (v-a)/(b-a)}
function drawRGB(frame){const c=$('rgb'),ctx=c.getContext('2d'),img=new Image();img.onload=()=>{c.width=img.naturalWidth;c.height=img.naturalHeight;ctx.drawImage(img,0,0)};img.src=frame.rgb}
function drawPCD(id,frame,axes){const c=$(id),ctx=c.getContext('2d'),w=720,h=720;c.width=w;c.height=h;ctx.fillStyle='#0b0f13';ctx.fillRect(0,0,w,h);const p=frame.points,b=data.bounds;let ax=axes[0],ay=axes[1];for(const q of p){const x=fit(q[ax],b[['x','y','z'][ax]][0],b[['x','y','z'][ax]][1]),y=fit(q[ay],b[['x','y','z'][ay]][0],b[['x','y','z'][ay]][1]);ctx.fillStyle=`rgb(${q[3]|0},${q[4]|0},${q[5]|0})`;ctx.globalAlpha=.78;ctx.beginPath();ctx.arc(24+x*(w-48),h-24-y*(h-48),2.2,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1}
function render(){const f=frames[i],s=f.stats||{};drawRGB(f);drawPCD('xy',f,[0,1]);drawPCD('xz',f,[0,2]);drawPCD('yz',f,[1,2]);slider.value=i;$('frameLabel').textContent=`frame ${i+1}/${frames.length} · step ${f.step}`;$('subtitle').textContent=`${data.task} · seed ${data.seed} · ${data.status}`;$('meta').innerHTML=[['foreground pixels',((s.foreground_pixel_fraction||0)*100).toFixed(1)+'%'],['sampled points',s.points||f.points.length],['object fraction',((s.object_fraction||0)*100).toFixed(1)+'%'],['finite depth',(s.finite_depth_fraction||0).toFixed(3)]].map(x=>`<div class="stat"><b>${x[1]}</b><span>${x[0]}</span></div>`).join('')}
slider.oninput=()=>{i=+slider.value;render()};$('prev').onclick=()=>{i=(i+frames.length-1)%frames.length;render()};$('next').onclick=()=>{i=(i+1)%frames.length;render()};render();
</script></body></html>""".replace("__PAYLOAD__", payload_json)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pen-v0", choices=("pen-v0", "relocate-v0"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = AdroitPCDConfig()
    env = _env_class(args.task)()
    env.seed(args.seed)
    env.reset()
    foreground, object_geom, labels = _geom_masks(env.sim.model)
    rng = np.random.default_rng(args.seed)
    frames = []
    for step in range(args.steps + 1):
        if step % args.stride == 0 or step == args.steps:
            frame = _capture(env.sim, config, foreground, object_geom, rng)
            frame["step"] = step
            frames.append(frame)
        if step < args.steps:
            action = np.clip(rng.normal(0.0, 0.15, env.action_space.shape), -1.0, 1.0)
            env.step(action)
    env.close()
    probe = {"task": args.task, "seed": args.seed, "frames": len(frames), "status": "GEOMETRY_PREVIEW_ONLY", "masks": labels}
    (args.out_dir / "probe.json").write_text(json.dumps(probe, indent=2), encoding="utf-8")
    _write_html(args.out_dir / "preview.html", args.task, args.seed, frames, labels)
    print(json.dumps({"preview": str(args.out_dir / "preview.html"), "probe": str(args.out_dir / "probe.json"), **probe}, indent=2))


if __name__ == "__main__":
    main()
