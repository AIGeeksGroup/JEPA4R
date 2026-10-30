"""One old EP0 fresh/repeat diagnostic; unchanged capture, no model or NN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(prepared_only=True, frames_if_run=2002, model_calls=0)))
        return
    import numpy as np
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    root = Path(plan['workspace']) / 'official_state_3d_cache_v1'
    sys.path.insert(0, str(root))
    import pcd_renderer as original
    expected_path = Path(plan['cache_dir']) / 'train/episode_000000/receipt.json'
    expected_bytes = expected_path.read_bytes()
    expected = json.loads(expected_bytes)
    os.setpriority(os.PRIO_PROCESS, 0, 10)
    original.init_worker(plan['workspace'], str(Path(plan['shared_state_root'])/'source_arrays.json'),
                         plan['camera'], plan['pcd'])
    original_capture = original._capture
    captures, images = [], None
    def wrapped_capture(frame, split, raw_id):
        cloud, rgb = original_capture(frame, split, raw_id)
        images[raw_id] = rgb
        item = frame.last_receipts['current']
        captures.append(dict(raw_id=raw_id, RGB_sha256=digest(rgb),
            PCD_sha256=digest(cloud), source_UV_sha256=item['source_UV_sha256'],
            source_geom_sha256=item['source_geom_sha256'],
            rng_before_sha256=item['rng_before_sha256'], rng_after_sha256=item['rng_after_sha256'],
            source_pose_sha256=item['source_pose_sha256'], source_time=item['source_time']))
        return cloud, rgb
    original._capture = wrapped_capture
    reports, first_images = [], None
    started = time.time()
    try:
        for label in ('fresh', 'repeat_same_worker'):
            captures = []
            images = np.empty((1001,224,224,3), np.uint8)
            value = original.render_episode(('train', 0))
            keys = ('sampling_seed','PCD_xyz_sha256','PCD_rgb_sha256','RGB_sha256',
                    'capture_receipt_stream_sha256','camera','render_geometry')
            record = dict(label=label, values={key:value[key] for key in keys},
                matches_original={key:value[key] == expected[key] for key in keys},
                per_frame=captures)
            if first_images is None:
                first_images = images
                prior_captures = captures
            else:
                delta = np.abs(images.astype(np.int16)-first_images.astype(np.int16))
                changing = np.flatnonzero(np.any(delta != 0, axis=(1,2,3)))
                record['fresh_repeat_image_comparison'] = dict(
                    bitexact=not len(changing), different_frames=len(changing),
                    different_channels=int(np.count_nonzero(delta)), max_abs_diff=int(delta.max()))
                if len(changing):
                    row = int(changing[0]); channel_positions = np.argwhere(delta[row] != 0)
                    samples = channel_positions[:32]
                    record['first_diff'] = dict(raw_id=row,
                        channel_coordinates=samples.tolist(),
                        fresh_values=[int(first_images[row,y,x,c]) for y,x,c in samples],
                        repeat_values=[int(images[row,y,x,c]) for y,x,c in samples])
                    np.savez_compressed(args.output_dir/'first_image_difference.npz',
                        raw_id=np.array(row), fresh=first_images[row], repeat=images[row])
                record['fresh_repeat_frame_metadata_matches'] = {
                    key:all(a[key] == b[key] for a,b in zip(prior_captures,captures))
                    for key in captures[0]}
            reports.append(record)
            del value
        assert expected_path.read_bytes() == expected_bytes
        model = original.STATE['original'].STATE['model']
        static_model = {key:digest(getattr(model,key)) for key in (
            'geom_rgba','mat_rgba','tex_data','light_pos','light_dir','cam_pos','cam_quat')
            if hasattr(model,key)}
        report = dict(actual_exit_code=0, diagnostic_completed=True,
            no_NN=True, no_production_cache_mutation=True, rendered_rows=2002,
            original_receipt_path=str(expected_path), original_receipt_sha256=hashlib.sha256(expected_bytes).hexdigest(),
            original_expected={key:expected[key] for key in keys}, results=reports,
            static_model_hashes=static_model, elapsed=time.time()-started)
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        (args.output_dir/'actual_exit_code').write_text('0\n')
    except BaseException as error:
        (args.output_dir/'failure.json').write_text(json.dumps(dict(error=repr(error),actual_exit_code=1))+'\n')
        (args.output_dir/'actual_exit_code').write_text('1\n')
        raise
    finally:
        original.STATE['original'].STATE['env'].close()


if __name__ == '__main__':
    main()
