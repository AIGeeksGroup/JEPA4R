"""The TriFinger CortexBench scores of run_tf_grid.sh: per run the best average scaled success on the 25 test demos over
the run's evaluations (trifinger_vc's diff-110_max_avg_scaled_reward, x 100), per task and encoder the seeds and their
mean; the number of evaluations a run has logged so far (reach: 21 when done, move: 41).
    STABLEWM_HOME=... python bench/cortexbench/summarize_tf.py"""
import json
import os
from pathlib import Path

RUNS = Path(os.environ['STABLEWM_HOME']) / 'logs' / 'cortexbench' / 'trifinger'
KEY = 'sim_env_demo/test/diff-110_max_avg_scaled_reward'
DONE = {'reach_cube': 21, 'move_cube': 41}
print('| task | encoder | seed 100 | seed 200 | seed 300 | mean |')
print('|---|---|---|---|---|---|')
for task in ('reach_cube', 'move_cube'):
    for rep in ('lewm_tf_pix', 'lewm_tf_pcd'):
        cells, vals = [], []
        for s in (100, 200, 300):
            f = RUNS / rep / task / f's{s}' / 'metrics.jsonl'
            recs = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
            if not recs:
                cells.append('-')
                continue
            v = 100 * recs[-1][KEY]
            vals.append(v)
            cells.append(f'{v:.1f}' + ('' if len(recs) >= DONE[task] else f' ({len(recs)}/{DONE[task]} evals)'))
        mean = f'{sum(vals) / len(vals):.1f}' if vals else '-'
        print(f'| {task} | {rep} | ' + ' | '.join(cells) + f' | {mean} |')
