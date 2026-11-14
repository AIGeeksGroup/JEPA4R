"""Command line of the Cube 1-4 paired collection: prints the plan, runs it with --execute."""

import argparse
import json

from . import build_plan, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cube-count", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--collection-seed", type=int, default=0)
    p.add_argument("--python")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    plan = build_plan(a.cube_count, a.output, a.python, collection_seed=a.collection_seed)
    print(json.dumps(plan, indent=2), flush=True)
    if a.execute:
        run(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
