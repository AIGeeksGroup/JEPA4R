"""Collect the grid's DONE files into per-task and per-suite scores, next to the VC-1 paper's Table 8.

Score per run, as CortexBench reports it: the best of the 20 evaluations over training (eval/highest_success for
Adroit and MetaWorld, in %; eval/highest_score / 10 for DMControl, whose returns are at most 1000 over 500 steps).
The last evaluation's value is listed too (no best-of-20 selection). Per task: mean over the seeds (seeds listed);
per suite: mean over its tasks, only where every task of the suite has runs.

    source env.sh && $PY summarize.py   -> prints the tables, writes $STABLEWM_HOME/logs/cortexbench/results/scores.{md,json}
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HOME = Path(os.environ['STABLEWM_HOME'])
RUNS = HOME / 'logs' / 'cortexbench' / 'runs'
OUT = HOME / 'logs' / 'cortexbench' / 'results'
TASKS = {  # env -> (suite, short name as in the paper's Table 8)
    'pen-v0': ('adroit', 'pen'), 'relocate-v0': ('adroit', 'relocate'),
    'assembly-v2-goal-observable': ('metaworld', 'assembly'), 'bin-picking-v2-goal-observable': ('metaworld', 'bin_picking'),
    'button-press-topdown-v2-goal-observable': ('metaworld', 'button_press'),
    'drawer-open-v2-goal-observable': ('metaworld', 'drawer_open'), 'hammer-v2-goal-observable': ('metaworld', 'hammer'),
    'dmc_finger_spin-v1': ('dmc', 'finger_spin'), 'dmc_reacher_easy-v1': ('dmc', 'reacher'),
    'dmc_cheetah_run-v1': ('dmc', 'cheetah_run'), 'dmc_walker_stand-v1': ('dmc', 'walker_stand'),
    'dmc_walker_walk-v1': ('dmc', 'walker_walk'),
}
ORDER = [short for _, short in TASKS.values()]
# VC-1 paper (arXiv 2303.18240v2) Table 8, per task (VIP / CLIP: the bin-picking entries only)
PAPER = {
    'vc1_vitb': dict(pen=76.0, relocate=32.0, assembly=88.0, bin_picking=78.7, button_press=82.7, drawer_open=100.0,
                     hammer=98.7, finger_spin=76.0, reacher=82.6, cheetah_run=32.3, walker_stand=83.5, walker_walk=44.7),
    'vc1_vitl': dict(pen=80.0, relocate=38.7, assembly=88.0, bin_picking=84.0, button_press=80.0, drawer_open=100.0,
                     hammer=92.0, finger_spin=76.8, reacher=88.0, cheetah_run=32.8, walker_stand=83.3, walker_walk=53.7),
    'r3m_rn50': dict(pen=81.3, relocate=65.3, assembly=97.3, bin_picking=93.3, button_press=89.3, drawer_open=100.0,
                     hammer=100.0, finger_spin=77.1, reacher=98.4, cheetah_run=66.1, walker_stand=93.8, walker_walk=70.1),
    'rand_vitb': dict(pen=4.0, relocate=0.0, assembly=0.0, bin_picking=0.0, button_press=2.7, drawer_open=0.0,
                      hammer=0.0, finger_spin=0.1, reacher=23.4, cheetah_run=0.4, walker_stand=22.7, walker_walk=4.0),
    'vip_rn50': dict(bin_picking=76.0),
    'clip_vitb16': dict(bin_picking=68.0),
}
LABELS = {
    'lewm_mw_button_pcd': 'LeWM 3D（点云，button-press 全量数据）',
    'lewm_mw_button_pix': 'LeWM 2D（图像，button-press 全量数据）',
    'lewm_mw_assembly_pcd': 'LeWM 3D（点云，assembly 全量数据）',
    'lewm_mw_assembly_pix': 'LeWM 2D（图像，assembly 全量数据）',
    'lewm_mw_binpick_pcd': 'LeWM 3D（点云，bin-picking 全量数据）',
    'lewm_mw_binpick_pix': 'LeWM 2D（图像，bin-picking 全量数据）',
    'lewm_mw_pcd': 'LeWM 3D（点云，bin-picking 小规模数据 15 万帧）',
    'lewm_mw_pix': 'LeWM 2D（图像，bin-picking 小规模数据 15 万帧）',
    'vc1_vitl': 'VC-1 ViT-L', 'vc1_vitb': 'VC-1 ViT-B', 'r3m_rn50': 'R3M RN-50', 'vip_rn50': 'VIP RN-50',
    'clip_vitb16': 'CLIP ViT-B/16', 'rand_vitb': '随机初始化 ViT-B（冻结）',
}

runs = defaultdict(lambda: defaultdict(list))  # model -> short task -> [(best, last, seed)]
for done in sorted(RUNS.glob('*/*/s*/DONE')):
    d = json.loads(done.read_text())
    if d['model'].startswith('_'):
        continue
    suite, short = TASKS[d['env']]
    scale = 10 if suite == 'dmc' else 1
    key_best, key_last = ('highest_score', 'final_score') if suite == 'dmc' else ('highest_success', 'final_success')
    runs[d['model']][short].append((d[key_best] / scale, d[key_last] / scale, d['seed']))

models = sorted(runs, key=lambda m: list(LABELS).index(m) if m in LABELS else len(LABELS))
tasks = [t for t in ORDER if any(t in runs[m] for m in models)]
res = {}
for m in models:
    res[m] = {}
    for t, vals in runs[m].items():
        vals = sorted(vals, key=lambda v: v[2])
        res[m][t] = dict(best=round(float(np.mean([v[0] for v in vals])), 1), last=round(float(np.mean([v[1] for v in vals])), 1),
                         seeds={int(v[2]): round(float(v[0]), 1) for v in vals}, paper=PAPER.get(m, {}).get(t))

lines = ['单个种子的分数 = 20 次评测中的最高值（Adroit / MetaWorld：成功率 %；DMC：回报 / 10），表中是种子均值，'
         '括号里是各种子；"末轮"= 最后一次评测的种子均值（不做 20 选 1）；"论文"= VC-1 论文表 8。', '']
for t in tasks:
    lines += [f'### {t}', '', '| 模型 | 种子 | 均值 | 论文 | 末轮 |', '|---|---|---|---|---|']
    for m in sorted((m for m in models if t in res[m]), key=lambda m: -res[m][t]['best']):
        r = res[m][t]
        lines.append(f"| {LABELS.get(m, m)} | {' / '.join(f'{v:g}' for v in r['seeds'].values())} | {r['best']} | "
                     f"{'—' if r['paper'] is None else r['paper']} | {r['last']} |")
    lines.append('')
suites = defaultdict(list)
for env, (suite, short) in TASKS.items():
    suites[suite].append(short)
suite_rows = []
for m in models:
    cells = {s: round(float(np.mean([res[m][t]['best'] for t in ts])), 1) for s, ts in suites.items() if all(t in res[m] for t in ts)}
    if cells:
        suite_rows.append(f"| {LABELS.get(m, m)} | " + ' | '.join(str(cells.get(s, '—')) for s in suites) + ' |')
if suite_rows:
    lines += ['### suite 均值（该 suite 每个任务都有结果时）', '', '| 模型 | Adroit | MetaWorld | DMControl |', '|---|---|---|---|'] + suite_rows
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
(OUT / 'scores.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
