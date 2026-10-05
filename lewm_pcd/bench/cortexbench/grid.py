"""Run CortexBench MuJoCo imitation jobs (official configs): models x tasks x seeds, one run_bc.py / run_bc3d.py each.

    source env.sh && mkdir -p $STABLEWM_HOME/logs/cortexbench && setsid nohup $PY grid.py \
        --models lewm_mw_binpick_pix,lewm_mw_binpick_pcd --tasks bin-picking [--seeds 100,200,300] [--slots 8] [--mps DIR] \
        >> $STABLEWM_HOME/logs/cortexbench/grid.log 2>&1 < /dev/null &

--tasks takes suites (adroit, metaworld, dmc) and tasks by short name (bin-picking, pen, reacher_easy, ...). A model whose
config (conf/model/<model>.yaml) calls lewm_cortex.load_lewm_pcd runs on point clouds: run_bc3d.py with the converted
MetaWorld demos (convert_demos_3d.py). Demos are read from $STABLEWM_HOME/datasets/cortexbench/<suite dir>. The suite
metaworld_rm75 (rm75-bin-picking, rm75-button-press-topdown, rm75-drawer-open, rm75-assembly: on the RM75-B + EG2-4C2,
envs/metaworld/rm75) runs run_bc_rm75.py, for both image and point-cloud models, on its own demos (make_rm75_demos.py);
its env runs in the python given by MW_PY.

Each job writes $STABLEWM_HOME/logs/cortexbench/runs/<model>/<env>/s<seed>/{metrics.jsonl, DONE} and s<seed>.log next to
it; a job whose DONE file exists is skipped, so the grid can be re-launched. On (re)launch, a run_bc process already
working on a job (started by an earlier grid.py) is adopted and waited for; a job whose metrics.jsonl is complete but has
no DONE file gets one; any other unfinished run dir is cleared and the job rerun. Concurrency: --slots jobs at a time; a
DMC job takes 2 slots and DMC jobs start at least DMC_GAP s apart: each loads a 9.2 GiB demo pickle and holds the frames
until they are embedded, so at most ~2 are at that peak at once. Jobs are niced. --mps DIR makes new jobs clients of the
CUDA MPS daemon whose pipe directory is DIR (their batch-1 encoder calls then share the GPU instead of time-slicing it).
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
RUNS = HOME / 'logs' / 'cortexbench' / 'runs'
VC_MODELS_CONF = HERE.parents[1] / 'third_party' / 'eai-vc' / 'vc_models' / 'src' / 'vc_models' / 'conf' / 'model'
SUITES = {  # suite -> (hydra config, demo dir, envs)
    'adroit': ('Adroit_BC_config', 'adroit-expert-v1.0', ['pen-v0', 'relocate-v0']),
    'metaworld': ('Metaworld_BC_config', 'metaworld-expert-v1.0', [
        'assembly-v2-goal-observable', 'bin-picking-v2-goal-observable', 'button-press-topdown-v2-goal-observable',
        'drawer-open-v2-goal-observable', 'hammer-v2-goal-observable']),
    'dmc': ('DMC_BC_config', 'dmc-expert-v1.0', [
        'dmc_finger_spin-v1', 'dmc_reacher_easy-v1', 'dmc_cheetah_run-v1', 'dmc_walker_stand-v1', 'dmc_walker_walk-v1']),
    'metaworld_rm75': ('Metaworld_BC_config', 'metaworld-rm75-expert', [
        'rm75-bin-picking-v2-goal-observable', 'rm75-button-press-topdown-v2-goal-observable',
        'rm75-drawer-open-v2-goal-observable', 'rm75-assembly-v2-goal-observable']),
}
RM75 = 'metaworld_rm75'
DMC_SLOTS, DMC_GAP = 2, 300

ap = argparse.ArgumentParser()
ap.add_argument('--models', required=True, help='comma list of embedding names')
ap.add_argument('--tasks', required=True, help='comma list of suites and/or task short names')
ap.add_argument('--seeds', default='100,200,300')
ap.add_argument('--slots', type=int, default=8)
ap.add_argument('--mps', default=None, help='CUDA_MPS_PIPE_DIRECTORY for new jobs')
args = ap.parse_args()


def log(*a):
    print(time.strftime('%F_%T'), *a, flush=True)


def resolve(name):
    """suite name -> its envs; short task name -> [(suite, env)]."""
    if name in SUITES:
        return [(name, e) for e in SUITES[name][2]]
    hits = [(s, e) for s, (_, _, envs) in SUITES.items() for e in envs
            if e in (name, f'{name}-v0', f'{name}-v2-goal-observable', f'dmc_{name}-v1')]
    if len(hits) != 1:
        sys.exit(f'unknown task {name!r}')
    return hits


def model_conf(model):
    for d in (HERE / 'conf' / 'model', VC_MODELS_CONF):
        if (d / f'{model}.yaml').exists():
            return yaml.safe_load((d / f'{model}.yaml').read_text())
    sys.exit(f'no config for embedding {model!r} in {HERE / "conf" / "model"} or {VC_MODELS_CONF}')


def evals_of(run):
    m = run / 'metrics.jsonl'
    return [json.loads(l) for l in m.read_text().splitlines() if '"eval/epoch"' in l] if m.exists() else []


def complete(evals):
    return bool(evals) and evals[-1]['eval/epoch'] >= 99


def write_done(job, evals, minutes):
    last = evals[-1]
    (job['run'] / 'DONE').write_text(json.dumps({**{k: job[k] for k in ('model', 'suite', 'env', 'seed')},
                                                  'minutes': round(minutes, 1),
                                                  'highest_success': last['eval/highest_success'],
                                                  'highest_score': last['eval/highest_score'],
                                                  'final_success': last['eval/success'],
                                                  'final_score': last['eval/score_mean'],
                                                  'n_evals': len(evals)}))
    log('done', job['model'], job['env'], job['seed'], f'{minutes:.1f} min',
        'highest_success', last['eval/highest_success'], 'highest_score', round(last['eval/highest_score'], 1))


def live_run_bc():
    """run dir -> pid of each live run_bc.py / run_bc3d.py process (zombies excluded)."""
    out = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / 'cmdline').read_bytes().split(b'\0')
            state = (p / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except OSError:
            continue
        if len(argv) > 1 and argv[1].endswith((b'run_bc.py', b'run_bc3d.py', b'run_bc_rm75.py')) and state != 'Z':
            for a in argv:
                if a.startswith(b'hydra.run.dir='):
                    out[a.split(b'=', 1)[1].decode()] = int(p.name)
    return out


def proc_start(pid):
    ticks = int((Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    btime = int(next(l for l in open('/proc/stat') if l.startswith('btime')).split()[1])
    return btime + ticks / os.sysconf('SC_CLK_TCK')


class Adopted:
    """A run_bc.py process started by an earlier grid.py: not our child, so no exit code; judged by its metrics."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            if (Path('/proc') / str(self.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                return None
        except OSError:
            pass
        self.returncode = 0
        return 0


tasks = [t for name in args.tasks.split(',') for t in resolve(name)]
models = args.models.split(',')
pcd = {m: model_conf(m).get('_target_') == 'lewm_cortex.load_lewm_pcd' for m in models}
jobs = []
for seed in map(int, args.seeds.split(',')):  # seed by seed: every finished seed is a complete comparison
    for model in models:
        for suite, env in tasks:
            if pcd[model] and suite not in ('metaworld', RM75):
                sys.exit(f'{model} runs on point clouds, which exist for MetaWorld only, not {env}')
            cfg, demos, _ = SUITES[suite]
            demos = DATA / (demos + '-pcd' if pcd[model] else demos)
            if not (demos / f'{env}.pickle').exists():
                sys.exit(f'missing demos {demos / env}.pickle (unpack_suite.py / convert_demos_3d.py / make_rm75_demos.py)')
            launcher = 'run_bc_rm75.py' if suite == RM75 else 'run_bc3d.py' if pcd[model] else 'run_bc.py'
            jobs.append(dict(model=model, suite=suite, env=env, seed=seed, cfg=cfg, demos=demos,
                             launcher=launcher, run=RUNS / model / env / f's{seed}'))
live = live_run_bc()
running = []  # (job, proc, slots, t0)
todo = []
for j in jobs:
    if (j['run'] / 'DONE').exists():
        continue
    if str(j['run']) in live:
        pid = live[str(j['run'])]
        running.append((j, Adopted(pid), DMC_SLOTS if j['suite'] == 'dmc' else 1, proc_start(pid)))
        log('adopted', j['model'], j['env'], j['seed'], 'pid', pid)
    elif complete(evals_of(j['run'])):  # finished while no grid.py was watching
        m = j['run'] / 'metrics.jsonl'
        write_done(j, evals_of(j['run']), (m.stat().st_mtime - (j['run'] / '.hydra').stat().st_mtime) / 60)
    else:
        todo.append(j)
if any(j['suite'] == RM75 for j in todo) and not os.environ.get('MW_PY'):
    sys.exit('the RM75 jobs need MW_PY, the lewm_mw python their env runs in (run_bc_rm75.py)')
log(f'{len(jobs)} jobs, {len(jobs) - len(todo) - len(running)} done already, {len(running)} adopted, {len(todo)} to run, '
    f'slots {args.slots}, mps {args.mps}')
env_vars = dict(os.environ, **({'CUDA_MPS_PIPE_DIRECTORY': args.mps} if args.mps else {}))
last_dmc_start = max([t0 for j, _, _, t0 in running if j['suite'] == 'dmc'], default=0.0)

while todo or running:
    for item in list(running):
        job, proc, slots, t0 = item
        if proc.poll() is None:
            continue
        running.remove(item)
        evals = evals_of(job['run'])
        if proc.returncode == 0 and complete(evals):
            write_done(job, evals, (time.time() - t0) / 60)
        else:
            log('FAILED', job['model'], job['env'], job['seed'], 'exit', proc.returncode, 'see', f"{job['run']}.log")
    used = sum(s for *_, s, _ in running)
    for job in list(todo):
        slots = DMC_SLOTS if job['suite'] == 'dmc' else 1
        if used + slots > args.slots or (job['suite'] == 'dmc' and time.time() - last_dmc_start < DMC_GAP):
            continue
        if job['run'].exists():  # an unfinished earlier attempt: start clean (metrics.jsonl is appended to)
            shutil.rmtree(job['run'])
        job['run'].mkdir(parents=True, exist_ok=True)
        cmd = ['nice', '-n', '10', sys.executable, str(HERE / job['launcher']), '--config-name', job['cfg'],
               f"env={job['env']}", f"embedding={job['model']}", f"seed={job['seed']}", f"data_dir={job['demos']}/",
               'wandb.project=local', 'wandb.entity=local', f"hydra.run.dir={job['run']}"]
        out = open(f"{job['run']}.log", 'w')
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=job['run'], env=env_vars)
        running.append((job, proc, slots, time.time()))
        todo.remove(job)
        if job['suite'] == 'dmc':
            last_dmc_start = time.time()
        used += slots
        log('start', job['model'], job['env'], job['seed'], f'(running {len(running)}, slots {used}/{args.slots})')
    time.sleep(20)
log('ALL_DONE')
