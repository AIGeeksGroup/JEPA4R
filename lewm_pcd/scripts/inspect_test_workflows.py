"""Inspect Test's 24 visual comparisons and separate State baseline without running a job."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.ogbench.test_workflows.catalog import STAGES, build_plan, load_catalog, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--list', action='store_true')
    actions.add_argument('--verify-sources', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--cube-count', type=int)
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if args.list or not any((args.model, args.cube_count, args.verify_sources)):
        value = {key: value for key, value in load_catalog().items() if key != 'groups'}
        entries = [dict(id=g['id'], model=g['model'], cube_count=g['cube_count'],
                        matrix=g['matrix']) for g in load_catalog()['groups']]
        value['groups'] = [g for g in entries if g['matrix'] == 'test24']
        value['baselines'] = [g for g in entries if g['matrix'] == 'state_baseline']
    elif args.verify_sources:
        value = verify_sources()
        if value['hash_mismatches']:
            print(json.dumps(value, indent=2))
            return 1
    else:
        if args.model is None or args.cube_count is None:
            parser.error('--model and --cube-count are required together')
        try:
            value = build_plan(args.model, args.cube_count, args.stage)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps(value, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
