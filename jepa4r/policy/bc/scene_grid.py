# Moved from cortexbench@9331494:lewm_pcd/bench/cortexbench/scene_grid.py (verbatim)
"""CortexBench imitation jobs on OGBench scene: encoder x task x seed, one run_bc_scene.py per job; a reduced grid.py
(the same command line, run directory layout and DONE files; no DMC / MPS parts, and no adoption of live jobs: a rerun
starts every run without DONE afresh, so an earlier scene_grid.py's jobs have to be stopped first).
    source env.sh && MW_PY=<lewm_mw python> [ENC_SOCKET=@lewm_scene_enc] setsid nohup $PY scene_grid.py \\
        --models lewm_scene_pix,lewm_scene_pcd [--tasks 1,2,3,4,5] [--seeds 100,200,300] [--slots 10] \\
        [--proprio hand6|none] >> $STABLEWM_HOME/logs/cortexbench/scene_grid.log 2>&1 &
Run directories $STABLEWM_HOME/logs/cortexbench/scene/runs_hand6/<encoder>/scene-task<k>-v0/s<seed>/{metrics.jsonl,
DONE}, with s<seed>.log beside them; apart from grid.py's runs/, which summarize.py reads as MetaWorld / Adroit / DMC
runs. --proprio: hand6 (default) = the first 6 proprioception entries (effector xyz, cos / sin yaw, gripper; not the
two locks); none = no proprioception, its runs under scene/runs_noproprio/. With ENC_SOCKET, enc_server.py must be
serving the --models.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ['STABLEWM_HOME'])
DATA = HOME / 'datasets' / 'cortexbench'
CONFS = [HERE / 'conf' / 'model']
START_GAP = 45  # s

ap = argparse.ArgumentParser()
ap.add_argument('--models', required=True)
ap.add_argument('--tasks', default='1,2,3,4,5')
ap.add_argument('--seeds', default='100,200,300')
ap.add_argument('--slots', type=int, default=10)
ap.add_argument('--proprio', choices=('hand6', 'none'), default='hand6')
args = ap.parse_args()
RUNS = HOME / 'logs' / 'cortexbench' / 'scene' / ('runs_hand6' if args.proprio == 'hand6' else 'runs_noproprio')
PROPRIO_ARGS = {'none': ['env_kwargs.add_proprio=False', 'env_kwargs.proprio_key=None'],
                'hand6': ['env_kwargs.add_proprio=True', 'env_kwargs.proprio_key=gripper_proprio']}[args.proprio]


def log(*a):
    print(time.strftime('%F_%T'), *a, flush=True)


def model_conf(m):
    for d in CONFS:
        if (d / f'{m}.yaml').exists():
            return yaml.safe_load((d / f'{m}.yaml').read_text())
    sys.exit(f'no config for {m}')


def evals_of(run):
    m = run / 'metrics.jsonl'
    return [json.loads(l) for l in m.read_text().splitlines() if '"eval/epoch"' in l] if m.exists() else []


def complete(evals):
    return bool(evals) and evals[-1]['eval/epoch'] >= 99


def write_done(job, evals, minutes):
    last = evals[-1]
    mean = sum(e['eval/success'] for e in evals) / len(evals)
    (job['run'] / 'DONE').write_text(json.dumps({**{k: job[k] for k in ('model', 'env', 'seed')}, 'minutes': round(minutes, 1),
                                                  'mean_success': mean, 'highest_success': last['eval/highest_success'],
                                                  'final_success': last['eval/success'], 'n_evals': len(evals)}))
    log('done', job['model'], job['env'], job['seed'], f'{minutes:.1f} min', 'mean_success', round(mean, 1),
        'final_success', last['eval/success'])


if not os.environ.get('MW_PY'):
    sys.exit('set MW_PY (the lewm_mw python the scene env runs in)')
models = args.models.split(',')
pcd = {m: model_conf(m).get('_target_') == 'lewm_cortex.load_lewm_pcd' for m in models}
jobs = []
for seed in map(int, args.seeds.split(',')):
    for m in models:
        for k in map(int, args.tasks.split(',')):
            env = f'scene-task{k}-v0'
            demos = DATA / ('scene-expert-pcd' if pcd[m] else 'scene-expert')
            if not (demos / f'{env}.pickle').exists():
                sys.exit(f'missing demos {demos}/{env}.pickle (make_scene_demos.py)')
            jobs.append(dict(model=m, env=env, seed=seed, demos=demos, run=RUNS / m / env / f's{seed}'))
todo = [j for j in jobs if not (j['run'] / 'DONE').exists()]
log(f'{len(jobs)} jobs, {len(jobs) - len(todo)} done already, {len(todo)} to run, slots {args.slots}')
running = []
while todo or running:
    for item in list(running):
        job, proc, t0 = item
        if proc.poll() is None:
            continue
        running.remove(item)
        evals = evals_of(job['run'])
        if proc.returncode == 0 and complete(evals):
            write_done(job, evals, (time.time() - t0) / 60)
        else:
            log('FAILED', job['model'], job['env'], job['seed'], 'exit', proc.returncode, 'see', f"{job['run']}.log")
    while todo and len(running) < args.slots:
        job = todo.pop(0)
        if job['run'].exists():
            shutil.rmtree(job['run'])
        job['run'].mkdir(parents=True, exist_ok=True)
        cmd = ['nice', '-n', '10', sys.executable, str(HERE / 'run_bc_scene.py'), '--config-name', 'Metaworld_BC_config',
               f"env={job['env']}", f"embedding={job['model']}", f"seed={job['seed']}", f"data_dir={job['demos']}/",
               'camera=front_pixels', *PROPRIO_ARGS, 'wandb.project=local', 'wandb.entity=local', f"hydra.run.dir={job['run']}"]
        if os.environ.get('ENC_SOCKET'):  # the encoder runs in enc_server.py; no CUDA here: BC and evaluation on the CPU
            cmd.append('device=cpu')
        out = open(f"{job['run']}.log", 'w')
        env = dict(os.environ)
        if args.proprio == 'hand6':
            env['SCENE_PROPRIO_DIMS'] = '6'
        if os.environ.get('ENC_SOCKET'):  # keep the job off the GPU (the DataLoader's pin_memory would open a CUDA context);
            env['CUDA_VISIBLE_DEVICES'] = ''  # run_bc_scene.py drops the variable again for the env subprocess (EGL rendering)
        running.append((job, subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=job['run'], env=env), time.time()))
        log('start', job['model'], job['env'], job['seed'], f'(running {len(running)})')
        time.sleep(START_GAP)  # staggered starts: each job first loads GBs of demos and encodes all frames; many at once fill the memory
    time.sleep(20)
log('ALL_DONE')
