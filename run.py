"""Dispatch the existing Cube and Relocate workflows without changing their plans."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lewm_pcd'))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--suite', choices=('cube', 'relocate'))
    selected, forwarded = parser.parse_known_args(argv)
    flags = {value.split('=', 1)[0] for value in forwarded}
    cube = bool(flags & {'--cube-count', '--cubes', '--config', '--rl-python'})
    relocate = bool(flags & {'--env', '--assets', '--sim-python'})
    if cube and relocate or selected.suite == 'cube' and relocate or selected.suite == 'relocate' and cube:
        parser.error('conflicting Cube and Relocate suite arguments')
    suite = selected.suite or ('cube' if cube else 'relocate' if relocate else None)
    if suite is None:
        if forwarded == ['--list']:
            from bench.ogbench.workflow import ALIASES, MODELS
            from lewm_pcd.bench.relocate.workflow import MODELS as RELOCATE
            print(json.dumps({'suites': {'cube': {'models': list(MODELS), 'aliases': ALIASES,
                                               'cube_counts': [1, 2, 3, 4]},
                                        'relocate': RELOCATE}}, indent=2))
            return 0
        if not forwarded or forwarded in (['--help'], ['-h']):
            parser.print_help()
            print('Use --suite cube|relocate, or retain the original --cube-count / --env flags.')
            return 0
        parser.error('choose --suite cube|relocate or provide original suite-specific arguments')
    if suite == 'cube':
        from bench.ogbench.workflow import main as workflow
    else:
        from lewm_pcd.bench.relocate.workflow import main as workflow
    return workflow(forwarded)


if __name__ == '__main__':
    raise SystemExit(main())
