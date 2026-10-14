"""Results table of the CortexBench imitation runs on OGBench scene (scene_grid.py): per encoder x task, a run's score
= the highest success rate of its 20 evaluations (CortexBench as is: the 20 are one training's evaluations every 5
epochs, the last at its end), averaged over the seeds; the last evaluation (epoch 99) is listed beside it.
    STABLEWM_HOME=<checkout>/lewm_pcd_data python summarize_scene.py [encoder ...] [--seeds 100,200,300] [--runs runs]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='*')
ap.add_argument('--seeds', default='100,200,300')
ap.add_argument('--runs', default='runs')
args = ap.parse_args()
RUNS = Path(os.environ['STABLEWM_HOME']) / 'logs' / 'cortexbench' / args.runs
SEEDS = [int(x) for x in args.seeds.split(',')]
models = args.models or sorted(p.name for p in RUNS.iterdir() if p.name.startswith('lewm_scene'))


def evals(m, k, s):
    f = RUNS / m / f'scene-task{k}-v0' / f's{s}' / 'metrics.jsonl'
    if not (f.parent / 'DONE').exists():
        return None
    return [json.loads(l)['eval/success'] for l in f.read_text().splitlines() if '"eval/epoch"' in l]


for title, fn in (('score: the highest of the 20 evaluations', np.max), ('the last evaluation', lambda v: v[-1])):
    print(f'\n{title}, mean over seeds {args.seeds}, %')
    print('| encoder | ' + ' | '.join(f'task {k}' for k in range(1, 6)) + ' | mean |')
    print('|---' * 7 + '|')
    for m in models:
        cells, means = [], []
        for k in range(1, 6):
            vals = [fn(e) for e in (evals(m, k, s) for s in SEEDS) if e]
            cells.append(f'{np.mean(vals):.1f}' + ('' if len(vals) == len(SEEDS) else f' ({len(vals)}/{len(SEEDS)})') if vals else '-')
            if len(vals) == len(SEEDS):
                means.append(np.mean(vals))
        print(f'| {m} | ' + ' | '.join(cells) + f' | {np.mean(means):.1f} |' if len(means) == 5 else f'| {m} | ' + ' | '.join(cells) + ' | - |')
