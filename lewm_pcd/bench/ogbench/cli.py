"""Describe a Cube command by default; execute only with an explicit flag."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

METHODS = ('pldm', 'lejepa', 'state', 'lewm_rgb', 'lewm_pcd', 'lewm2d', 'lewm3d')
METHOD_ALIASES = {'lewm2d': 'lewm_rgb', 'lewm3d': 'lewm_pcd'}
STAGES = ('train', 'gciql', 'native-sr')
REPO = Path(__file__).resolve().parents[3]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--list', action='store_true', help='list profiles and supported methods without framework imports')
    result.add_argument('--profile')
    result.add_argument('--cube-count', type=int, choices=(1, 2, 3, 4))
    result.add_argument('--method', choices=METHODS)
    result.add_argument('--stage', choices=STAGES)
    result.add_argument('--machine-profile', type=Path, help='JSON containing local paths, Python and I/O resources')
    result.add_argument('--data-root', type=Path)
    result.add_argument('--output-root', type=Path)
    result.add_argument('--config', type=Path, help='frozen downstream configuration or explicit LeWM recipe')
    result.add_argument('--python')
    result.add_argument('--workers', type=int)
    result.add_argument('--prefetch', type=int)
    result.add_argument('--execute', action='store_true', help='run the described local child command')
    result.add_argument('--report', type=Path, help='save a new command-plan JSON; never overwrite')
    return result


def resolve_options(args):
    if args.machine_profile:
        values = json.loads(args.machine_profile.read_text(encoding='utf8'))
        allowed = {'data_root', 'output_root', 'python', 'workers', 'prefetch'}
        if set(values) - allowed:
            raise ValueError('unknown machine-profile fields: ' + ', '.join(sorted(set(values) - allowed)))
        for key, value in values.items():
            if getattr(args, key) is None:
                setattr(args, key, Path(value) if key.endswith('_root') else value)
    for key in ('profile', 'method', 'stage', 'data_root', 'output_root'):
        if getattr(args, key) is None:
            raise ValueError('--' + key.replace('_', '-') + ' is required')
    for key in ('workers', 'prefetch'):
        value = getattr(args, key)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(key + ' must be a positive integer')
    return args


def build_plan(args):
    from .contracts import get_profile
    from . import training
    args = resolve_options(args)
    requested_method = args.method
    args.method = METHOD_ALIASES.get(args.method, args.method)
    profile = get_profile(args.profile)
    if args.cube_count is not None and args.cube_count != profile['cube_count']:
        raise ValueError('cube count disagrees with the explicit data profile')
    if args.stage == 'train':
        if args.method not in ('pldm', 'lejepa', 'lewm_rgb', 'lewm_pcd'):
            raise ValueError('the unified train stage supports PLDM, Tiny LeJEPA and bound LeWM RGB/PCD recipes')
        command = training.build_command(args.profile, args.method, args.data_root, args.output_root,
                                         workers=args.workers, prefetch=args.prefetch, python=args.python,
                                         config_path=args.config)
    else:
        if args.config is None:
            raise ValueError('--config is required for downstream stages')
        from .downstream import build_command
        command = build_command(args.profile, args.method, args.stage, args.data_root,
                                args.output_root, args.config, python=args.python)
    environment = training.build_environment(args.data_root, args.output_root)
    wm_binding = None
    if args.stage == 'train' and args.method in ('lewm_rgb', 'lewm_pcd'):
        from .lewm_data import describe
        wm_binding = describe(args.profile, '3d' if args.method == 'lewm_pcd' else '2d', args.data_root)
    return dict(schema='cube_unified_command_v1', execute=args.execute, profile=profile,
                requested_method=requested_method, method=args.method, stage=args.stage, command=[str(item) for item in command],
                environment={key: environment[key] for key in ('PYTHONPATH', 'STABLEWM_HOME')},
                old_entry_points_preserved=True, lejepa_recipe='cortexbench_tiny' if args.method == 'lejepa' else None,
                wm_training_data_binding=wm_binding,
                profile_role='task_selector; LeWM inputs are wm_training_data_binding' if wm_binding else 'dataset_profile',
                lewm_recipe=({'encoder': 'DexArtPointNet' if args.method == 'lewm_pcd' else 'ViTCLS_tiny_patch14',
                              'latent_dim': 192, 'num_steps': 4, 'frameskip': 5, 'action_input_dim': 25}
                             if args.stage == 'train' and args.method in ('lewm_rgb', 'lewm_pcd') else None))


def verify_source_manifest():
    path = REPO / 'UNIFIED_SOURCE_MANIFEST.json'
    value = json.loads(path.read_text(encoding='utf8'))
    for name, expected in value['files'].items():
        candidate = (REPO / name).resolve()
        if not candidate.is_relative_to(REPO) or hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise ValueError('frozen source differs: ' + name)
    return value


def execute(plan, args):
    from .training import build_environment
    verify_source_manifest()
    output = args.output_root.resolve()
    dispatch = output.with_name(output.name + '.dispatch')
    if output.exists() or dispatch.exists():
        raise FileExistsError('fresh output and dispatch directories are required')
    dispatch.mkdir(parents=True)
    def stamp():
        return datetime.now(timezone.utc).isoformat()
    state = dict(stage='running', started_at=stamp(), updated_at=stamp(),
                 config=plan, completed_count=0, failed_count=0, actual_exit_code=None)
    def save():
        temporary = dispatch / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
        temporary.replace(dispatch / 'status.json')
    save()
    (dispatch / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf8')
    try:
        with (dispatch / 'stdout_stderr.log').open('x', encoding='utf8') as log:
            result = subprocess.run(plan['command'], env=build_environment(args.data_root, output),
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        code = result.returncode
        state.update(stage='child_exited', actual_exit_code=code,
                     completed_count=int(code == 0), failed_count=int(code != 0))
    except BaseException as error:
        code = 1
        state.update(stage='dispatch_failed', actual_exit_code=code, failed_count=1, error=repr(error))
    state['updated_at'] = state['finished_at'] = stamp()
    save()
    (dispatch / 'actual_exit_code').write_text(str(code) + '\n', encoding='utf8')
    return code


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.list:
        from .contracts import profile_names
        print(json.dumps(dict(profiles=list(profile_names()), methods=list(METHODS), stages=list(STAGES))))
        return 0
    try:
        plan = build_plan(args)
        if args.report:
            with args.report.open('x', encoding='utf8') as stream:
                json.dump(plan, stream, indent=2)
                stream.write('\n')
        print(json.dumps(plan, indent=2))
        return execute(plan, args) if args.execute else 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        ap.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
