"""Eight same-pose captures and existing-only pilot/full boundary comparison."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    args.output_dir.mkdir(exist_ok=False)
    plan = json.loads(args.plan.read_text())
    root = Path(plan['workspace'])/'official_state_3d_cache_v1'
    sys.path.insert(0,str(root))
    import pcd_renderer as renderer
    cache = Path(plan['cache_dir'])
    boundaries = {}
    for split in ('train','val'):
        path = cache/f'pilot_{split}_boundary.npz'
        with np.load(path,allow_pickle=False) as old:
            ids = old['raw_ids']; index = int(np.flatnonzero(ids == 1001)[0])
            xyz, rgb, z = old['pcd_xyz'][index],old['pcd_rgb'][index],old['latents'][index]
        receipt_path = cache/split/'episode_000001/receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        report = dict(pilot_path=str(path),raw_id=1001,
            pilot_cloud_sha256=sha(np.concatenate((xyz,rgb.astype(np.float32)),axis=1)),
            pilot_latent_sha256=sha(z), full_EP1_committed=receipt is not None)
        if receipt:
            latent = np.load(receipt['latent']['path'],mmap_mode='r',allow_pickle=False)[0]
            report.update(full_cloud_sha256=receipt['capture_boundaries']['0']['cloud_sha256'],
                full_latent_sha256=sha(latent),
                cloud_bitexact=report['pilot_cloud_sha256']==receipt['capture_boundaries']['0']['cloud_sha256'],
                latent_bitexact=bool(np.array_equal(z,latent)),
                latent_max_abs_diff=float(np.max(np.abs(z-latent))),
                full_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest())
        boundaries[split] = report
    renderer.init_worker(plan['workspace'],str(Path(plan['shared_state_root'])/'source_arrays.json'),
                         plan['camera'],plan['pcd'])
    frame = renderer._frame('train',0)
    original = frame.cores['current']._point_cloud
    state = copy.deepcopy(frame.cores['current']._rng.bit_generator.state)
    trace = {}
    def capture_trace(**kwargs):
        result = original(**kwargs)
        trace.update(uv=result['pcd_trace']['source_pixel_uv'].copy())
        return result
    frame.cores['current']._point_cloud = capture_trace
    pictures, results = [], []
    try:
        for trial in range(8):
            frame.cores['current']._rng.bit_generator.state = copy.deepcopy(state)
            cloud, rgb = renderer._capture(frame,'train',5)
            pictures.append(rgb.copy())
            seg = frame.provider.last_seg[...,0]
            results.append(dict(trial=trial,RGB_sha256=sha(rgb),cloud_sha256=sha(cloud),
                seg_sha256=sha(seg),UV_sha256=sha(trace['uv']),
                geometry_source_hash=frame.last_receipts['current']['source_pose_sha256']))
        reference = pictures[0]
        mismatch = np.any(np.stack(pictures) != reference,axis=(0,3))
        positions = np.argwhere(mismatch)
        groups = {}
        core, model = frame.cores['current'],renderer.STATE['original'].STATE['model']
        for y,x in positions:
            geom = int(seg[y,x])
            if geom < 0:
                label = 'sky_no_geometry'
            elif not core._geom_fg[geom]:
                label = 'excluded_world_or_mocap'
            elif core._geom_cube[geom]:
                label = 'cube_foreground'
            else:
                label = 'robot_foreground'
            groups[label] = groups.get(label,0)+1
        uvset = {(int(y),int(x)) for x,y in trace['uv']}
        report = dict(actual_exit_code=0,rendered_rows=8,neural_calls=0,
            old_cache_changed=False,raw_id=5,
            RNG_note='All eight captures reset the same EP0 private RNG; diagnostic is not original t5 stream replay',
            results=results,all_cloud_bitexact=len({x['cloud_sha256'] for x in results})==1,
            all_seg_bitexact=len({x['seg_sha256'] for x in results})==1,
            all_UV_bitexact=len({x['UV_sha256'] for x in results})==1,
            RGB_changed_pixels=int(mismatch.sum()),RGB_changed_pixel_classes=groups,
            differing_pixels_sampled_by_cloud=sum((int(y),int(x)) in uvset for y,x in positions),
            max_RGB_channel_abs_diff=int(np.max(np.abs(np.stack(pictures).astype(np.int16)-reference.astype(np.int16)))),
            sample_pixel_changes=[dict(y=int(y),x=int(x),geom=int(seg[y,x]),
                values=[image[y,x].tolist() for image in pictures]) for y,x in positions[:32]],
            old_pilot_full_boundary=boundaries,
            model_texture_sha256=sha(model.tex_data),
            hypothesis_status='same-pose repeated AA4 output measured; shader or skybox root cause not assumed')
        if len(positions):
            np.savez_compressed(args.output_dir/'raw5_images_diagnostic.npz',
                RGB=np.stack(pictures),seg=seg,UV=trace['uv'])
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'actual_exit_code').write_text('0\n')
        print(json.dumps(report))
    except BaseException as error:
        (args.output_dir/'failure.json').write_text(json.dumps(dict(error=repr(error),actual_exit_code=1))+'\n')
        (args.output_dir/'actual_exit_code').write_text('1\n')
        raise
    finally:
        frame.close()
        renderer.STATE['original'].STATE['env'].close()


if __name__ == '__main__':
    main()
