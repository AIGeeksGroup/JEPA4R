"""The DQC table of the puzzle boards: each input's success rate (OGBench's 5 tasks x 50 episodes, %) by OGBench's
measure (the mean of the evaluations at 800 k, 900 k and 1 M steps) and the highest evaluation of the run (every
100 k steps), from logs/dqc/puzzle_<board>/<input>/eval.jsonl.
    python summarize_dqc.py [--boards 3x3 4x4 4x5]
"""
import argparse
import json
import os
from pathlib import Path

INPUTS = [('lewm2d', 'LeWM 2D'), ('lewm3d', 'LeWM 3D'), ('vjepa21', 'V-JEPA 2.1 (frozen)'), ('visreg', 'VISReg (frozen)'),
          ('lejepa', 'LeJEPA'), ('pldm', 'PLDM')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', nargs='+', default=['3x3', '4x4', '4x5'])
    a = ap.parse_args()
    runs = Path(os.environ['STABLEWM_HOME']) / 'logs' / 'dqc'
    print('success %: OGBench measure (the mean at 800 k / 900 k / 1 M steps) / the highest evaluation')
    print('| input | ' + ' | '.join(a.boards) + ' |')
    print('|---|' + '---|' * len(a.boards))
    for key, name in INPUTS:
        cells = []
        for b in a.boards:
            f = runs / f'puzzle_{b}' / key / 'eval.jsonl'
            if not f.exists():
                cells.append('–')
                continue
            ev = {r['step']: 100 * r['overall_success'] for r in map(json.loads, f.read_text().splitlines())}
            last = [ev[s] for s in (800000, 900000, 1000000) if s in ev]
            ogb = f'{sum(last) / 3:.1f}' if len(last) == 3 else f'(at {max(ev) // 1000} k)'
            cells.append(f'{ogb} / {max(ev.values()):.1f}')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
