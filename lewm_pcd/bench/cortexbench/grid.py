"""Run CortexBench MuJoCo imitation jobs (official configs): models x tasks x seeds, one run_bc.py / run_bc3d.py each.

    source env.sh && mkdir -p $STABLEWM_HOME/logs/cortexbench && setsid nohup $PY grid.py \
        --models lewm_mw_binpick_pix,lewm_mw_binpick_pcd --tasks bin-picking [--seeds 100,200,300] [--slots 8] [--mps DIR] \
        >> $STABLEWM_HOME/logs/cortexbench/grid.log 2>&1 < /dev/null &

--tasks takes suites (adroit, metaworld, dmc) and tasks by short name (bin-picking, pen, reacher_easy, ...). A model whose
config (conf/model/<model>.yaml) calls lewm_cortex.load_lewm_pcd runs on point clouds: run_bc3d.py with the converted
MetaWorld demos (convert_demos_3d.py). Demos are read from $STABLEWM_HOME/datasets/cortexbench/<suite dir>.

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
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from .protocol import build_job_contract, expected_evaluator_epochs, num_demos_for_suite, validate_metric_records
except ImportError:  # script execution from this directory
    from protocol import build_job_contract, expected_evaluator_epochs, num_demos_for_suite, validate_metric_records

HERE = Path(__file__).resolve().parent
HOME = Path(os.environ['STABLEWM_HOME'])
DATA = HOME / 'datasets' / 'cortexbench'
VC_MODELS_CONF = HERE.parents[1] / 'third_party' / 'eai-vc' / 'vc_models' / 'src' / 'vc_models' / 'conf' / 'model'
SUITES = {  # suite -> (hydra config, demo dir, envs)
    'adroit': ('Adroit_BC_config', 'adroit-expert-v1.0', ['pen-v0', 'relocate-v0']),
    'metaworld': ('Metaworld_BC_config', 'metaworld-expert-v1.0', [
        'assembly-v2-goal-observable', 'bin-picking-v2-goal-observable', 'button-press-topdown-v2-goal-observable',
        'drawer-open-v2-goal-observable', 'hammer-v2-goal-observable']),
    'dmc': ('DMC_BC_config', 'dmc-expert-v1.0', [
        'dmc_finger_spin-v1', 'dmc_reacher_easy-v1', 'dmc_cheetah_run-v1', 'dmc_walker_stand-v1', 'dmc_walker_walk-v1']),
}
DMC_SLOTS, DMC_GAP = 2, 300

ap = argparse.ArgumentParser()
ap.add_argument('--models', required=True, help='comma list of embedding names')
ap.add_argument('--tasks', required=True, help='comma list of suites and/or task short names')
ap.add_argument('--seeds', default='100,200,300')
ap.add_argument('--slots', type=int, default=8)
ap.add_argument('--mps', default=None, help='CUDA_MPS_PIPE_DIRECTORY for new jobs')
ap.add_argument('--data-root', default=str(DATA), help='root containing unpacked official suite demo directories')
ap.add_argument('--output-root', default=str(HOME / 'logs' / 'cortexbench' / 'runs'),
                help='独立运行输出根；旧结果默认保留在原路径')
args = ap.parse_args()

DATA = Path(args.data_root).resolve()
RUNS = Path(args.output_root).resolve()


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


def complete(evals, epochs=100, eval_frequency=5):
    return len(evals) == len(expected_evaluator_epochs(epochs, eval_frequency)) and [
        int(float(e['eval/epoch'])) for e in evals
    ] == expected_evaluator_epochs(epochs, eval_frequency)


def validate_job(job):
    """Validate a finished official run before creating its DONE marker."""
    metrics_path = job['run'] / 'metrics.jsonl'
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    summary = validate_metric_records(records, job['epochs'], job['eval_frequency'])
    config_path = job['run'] / '.hydra' / 'config.yaml'
    if not config_path.exists():
        raise ValueError(f'missing resolved config: {config_path}')
    resolved = yaml.safe_load(config_path.read_text())
    for key, expected in {
        'env': job['env'], 'num_demos': job['num_demos'], 'epochs': job['epochs'],
        'eval_frequency': job['eval_frequency'], 'eval_num_traj': job['eval_num_traj'],
    }.items():
        if resolved.get(key) != expected:
            raise ValueError(f'resolved config mismatch for {key}: {resolved.get(key)!r} != {expected!r}')
    policies = list(job['run'].rglob('best_policy.pickle'))
    if len(policies) != 1:
        raise ValueError(f'expected one best_policy.pickle, found {len(policies)}')
    return summary, resolved, policies[0]


def write_done(job, evals, minutes):
    summary, resolved, policy = validate_job(job)
    (job['run'] / 'DONE').write_text(json.dumps({**{k: job[k] for k in ('model', 'suite', 'env', 'seed')},
                                                  'num_demos': job['num_demos'],
                                                  'epochs': job['epochs'],
                                                  'eval_frequency': job['eval_frequency'],
                                                  'eval_num_traj': job['eval_num_traj'],
                                                  'data_dir': str(job['demos']),
                                                  'result_scope': job['result_scope'],
                                                  'policy_path': str(policy),
                                                  'resolved_config': resolved,
                                                  'minutes': round(minutes, 1),
                                                  'highest_success': summary['best_success'],
                                                  'highest_score': summary['best_score'],
                                                  'final_success': summary['final_success'],
                                                  'final_score': summary['final_score'],
                                                  'n_evals': summary['eval_records']}))
    log('done', job['model'], job['env'], job['seed'], f'{minutes:.1f} min',
        'highest_success', summary['best_success'], 'highest_score', round(summary['best_score'], 1))


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
        if len(argv) > 1 and argv[1].endswith((b'run_bc.py', b'run_bc3d.py')) and state != 'Z':
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
            if pcd[model] and suite != 'metaworld':
                sys.exit(f'{model} runs on point clouds, which exist for MetaWorld only, not {env}')
            cfg, demos, _ = SUITES[suite]
            demos = DATA / (demos + '-pcd' if pcd[model] else demos)
            if not (demos / f'{env}.pickle').exists():
                sys.exit(f'missing demos {demos / env}.pickle (unpack_suite.py / convert_demos_3d.py)')
            job = build_job_contract(suite, env, model, seed, str(demos), str(RUNS / model / env / f's{seed}'), pcd[model])
            job.update(cfg=cfg, launcher='run_bc3d.py' if pcd[model] else 'run_bc.py', run=RUNS / model / env / f's{seed}', demos=demos)
            jobs.append(job)
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
    elif complete(evals_of(j['run']), j['epochs'], j['eval_frequency']):  # finished while no grid.py was watching
        m = j['run'] / 'metrics.jsonl'
        write_done(j, evals_of(j['run']), (m.stat().st_mtime - (j['run'] / '.hydra').stat().st_mtime) / 60)
    elif j['run'].exists() and any(j['run'].iterdir()):
        sys.exit(f'refusing to overwrite incomplete run: {j["run"]}')
    else:
        todo.append(j)
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
        if proc.returncode == 0 and complete(evals, job['epochs'], job['eval_frequency']):
            try:
                write_done(job, evals, (time.time() - t0) / 60)
            except (OSError, ValueError, KeyError) as exc:
                log('FAILED_VALIDATION', job['model'], job['env'], job['seed'], repr(exc))
                raise SystemExit(1)
        else:
            log('FAILED', job['model'], job['env'], job['seed'], 'exit', proc.returncode, 'see', f"{job['run']}.log")
            raise SystemExit(1)
    used = sum(s for *_, s, _ in running)
    for job in list(todo):
        slots = DMC_SLOTS if job['suite'] == 'dmc' else 1
        if used + slots > args.slots or (job['suite'] == 'dmc' and time.time() - last_dmc_start < DMC_GAP):
            continue
        if job['run'].exists() and any(job['run'].iterdir()):
            raise SystemExit(f'refusing to overwrite existing run: {job["run"]}')
        job['run'].mkdir(parents=True, exist_ok=True)
        (job['run'] / 'resolved_job.json').write_text(json.dumps(job, indent=2, default=str))
        cmd = ['nice', '-n', '10', sys.executable, str(HERE / job['launcher']), '--config-name', job['cfg'],
               f"env={job['env']}", f"embedding={job['model']}", f"seed={job['seed']}", f"data_dir={job['demos']}/",
               f"num_demos={job['num_demos']}", f"epochs={job['epochs']}",
               f"eval_frequency={job['eval_frequency']}", 'save_frequency=10',
               f"eval_num_traj={job['eval_num_traj']}", 'num_cpu=1',
               'device=cpu', 'data_parallel=false',
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
