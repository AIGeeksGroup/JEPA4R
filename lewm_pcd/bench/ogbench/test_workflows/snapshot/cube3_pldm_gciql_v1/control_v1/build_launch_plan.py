"""Build one exclusive phase plan only from frozen component handoffs; never execute it."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from serial import SOURCE_SHA, GUARD_SHA

ROOT = '/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1'
CASE = ROOT+'/cube3_pldm_gciql_v1'
GPU_LEASE = ROOT+'/cube4_visual_gciql_v1/control_v1/gpu_lease.py'
GPU_LEASE_SHA = 'd3a1757ec4b22746da404f15cea2f99e7e830c1edf0f07e1ac20a96955a13727'
PHASES = ('00_prepare', '01_cache_pilot', '02_cache_full', '03_cache_audit', '04_prepare_train',
    '05_GCIQL_gate', '06_GCIQL_reload', '07_GCIQL_train', '08_GCIQL_accept',
    '09_prepare_native', '10_native_tiny', '11_native_tiny_audit', '12_native_full', '13_native_full_audit')


def remote_path(value):
    path = PurePosixPath(value)
    assert isinstance(value, str) and '\\' not in value and path.is_absolute()
    assert '..' not in path.parts and path.is_relative_to(PurePosixPath(ROOT))
    return value


def build(bindings_path):
    blob = Path(bindings_path).read_bytes(); binding = json.loads(blob)
    assert binding['schema_version'] == 1 and binding['case'] == CASE
    assert binding['all_component_CLIs_and_sources_frozen'] is True
    assert [p['name'] for p in binding['phases']] == list(PHASES)
    components = []
    for component in binding['frozen_component_handoffs']:
        source = Path(component['local_path']); raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == component['sha256']
        json.loads(raw)
        components.append(dict(path=str(source.resolve()), bytes=len(raw), sha256=component['sha256']))
    assert components
    pins = dict(binding['source_pins'])
    pins[ROOT+'/cube2_official_embeddings_v1/serial_relay.py'] = SOURCE_SHA
    pins[ROOT+'/prepare_runtime.py'] = GUARD_SHA
    pins[GPU_LEASE] = GPU_LEASE_SHA
    frozen_path = Path(__file__).with_name('SOURCE_FREEZE_v1.json')
    frozen = json.loads(frozen_path.read_bytes())
    for name, item in frozen['files'].items():
        data = (frozen_path.parent/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        pins[CASE+'/control_v1/'+name] = item['sha256']
    pins[CASE+'/control_v1/SOURCE_FREEZE_v1.json'] = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    for path, digest in pins.items():
        remote_path(path); assert len(bytes.fromhex(digest)) == 32
        assert PurePosixPath(path).suffix not in ('.npy', '.npz', '.h5', '.hdf5')
    for phase in binding['phases']:
        assert phase['argv'] and all(isinstance(x, str) and x for x in phase['argv'])
        assert not any('TEMPLATE' in value for value in phase['argv'])
        remote_path(phase['cwd']); assert phase['cwd'] == ROOT
        assert isinstance(phase['environment'], dict) and all(isinstance(k, str) and isinstance(v, str) for k,v in phase['environment'].items())
        assert 'CUDA_VISIBLE_DEVICES' in phase['environment'] and 'JAX_PLATFORMS' in phase['environment']
        if phase['environment']['CUDA_VISIBLE_DEVICES']:
            assert phase['environment']['CUDA_VISIBLE_DEVICES'] == '0'
            phase['original_child_argv'] = list(phase['argv'])
            phase['argv'] = [ROOT+'/env_runtime/bin/python', '-B', GPU_LEASE, '--', *phase['argv']]
            phase['GPU_empty_guard_same_PGID_exec'] = True
        assert phase['checks'] and any(c['equals'].get('actual_exit_code') == 0 or c['equals'].get('exit_code') == 0 for c in phase['checks'])
        for check in phase['checks']:
            remote_path(check['path']); assert check['equals']
            assert PurePosixPath(check['path']).suffix == '.json'
    return dict(schema_version=1, cube_count=3, data_id='cube-triple-play-v0',
        training_updates=1000000, SR_episodes=250,
        pipeline_budget=dict(new_GCIQL_updates=1000000, formal_Native_episodes=250,
            diagnostic_Native_steps=4, diagnostic_excluded_from_SR=True),
        process_guard=ROOT+'/prepare_runtime.py', own_RSS_limit_bytes=24*1024**3,
        source_pins=pins, phases=binding['phases'],
        component_handoffs=components, bindings_sha256=hashlib.sha256(blob).hexdigest(),
        no_automatic_restart_or_fallback=True, all_phases_serial=True,
        encoder_dim=512, canonical_features=3303300,
        generator=dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); plan = build(args.bindings)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(prepared_only=True, output=str(args.output), phases=len(plan['phases']), no_job_started=True)))
