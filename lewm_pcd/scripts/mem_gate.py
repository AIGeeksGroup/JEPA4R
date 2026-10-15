"""Memory / GPU admission for the pipelines' long jobs, and a watchdog that keeps the container responsive.

The container (cgroup v2): above memory.high (88 GiB) every process is throttled and the machine stops responding; at
memory.max (90 GiB) the largest process is killed. Page cache is dropped by TMP/ops/cache_guard.py; what cannot be
dropped is anon + shmem (memory.stat), "fixed" below.

  admit     start a job (detached: its own session, stdin /dev/null, output appended to --log, pid in --pidfile) once
            fixed + the reservations still pending + its --mem <= BUDGET and the GPU has --gpu MiB free; a job's
            reservation stays pending until its own processes use it (max(0, mem - used)). The check and the start run
            under the pipelines' gate lock (logs/train/.gpu_gate.lock), which is held RAMP s more after a start so the
            GPU reading includes the new job. Blocks until the job is started.
  register  add a job started otherwise (already running) to the registry, so the watchdog can pause it.
  watch     every POLL s: fixed above PAUSE: the newest running registered job is paused (SIGSTOP to its process
            group), the next newest one PAUSE_STEP s later if still above; above KILL: the newest one gets SIGTERM
            (DQC / GCIQL save and exit on it); below RESUME the paused jobs are continued in the reverse order (the
            older jobs first, the newest last), one per PAUSE_STEP s. On its start it continues every registered job
            (none is left stopped by an earlier watchdog).
  status    the registry's live jobs with their memory now.
Registry: logs/train/mem_gate.jsonl, one line per job (name, pid, its start time, mem, gpu).
    python scripts/mem_gate.py admit --name puzzle_4x4_pix --mem 30 --gpu 15000 --pidfile <f> --log <f> [--cwd <dir>] \\
        -- cmd args...
    setsid nohup python scripts/mem_gate.py watch > $STABLEWM_HOME/logs/mem_gate_watch.log 2>&1 < /dev/null &
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time

BUDGET, PAUSE, KILL, RESUME = 72.0, 80.0, 85.0, 70.0  # GiB of fixed memory; memory.high is 88
POLL, PAUSE_STEP, RAMP = 5.0, 30.0, 600.0
HOME = os.environ.get('STABLEWM_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                                                    'lewm_pcd_data'))
REG = os.path.join(HOME, 'logs', 'train', 'mem_gate.jsonl')
LOCK = os.path.join(HOME, 'logs', 'train', '.gpu_gate.lock')


def now():
    return time.strftime('%F_%T')


def fixed_gib():
    s = dict(line.split() for line in open('/sys/fs/cgroup/memory.stat'))
    return (int(s['anon']) + int(s['shmem'])) / 2**30


def free_gpu_mib():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def start_time(pid):
    """The process's start time (clock ticks since boot): with the pid it identifies the process (pids are reused);
    None once it has exited (a zombie included)."""
    try:
        f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
        return None if f[0] == 'Z' else int(f[19])
    except (OSError, IndexError, ValueError):
        return None


def live_jobs():
    """Registered jobs whose process is still the registered one (newest last)."""
    jobs = []
    if os.path.exists(REG):
        for line in open(REG):
            j = json.loads(line)
            if start_time(j['pid']) == j['start']:
                jobs.append(j)
    return jobs


def group_gib(pgid):
    """Memory (Pss anon + shmem) of the processes in the process group."""
    tot = 0
    for p in os.listdir('/proc'):
        if not p.isdigit():
            continue
        try:
            if os.getpgid(int(p)) != pgid:
                continue
            for line in open(f'/proc/{p}/smaps_rollup'):
                if line.startswith(('Pss_Anon:', 'Pss_Shmem:')):
                    tot += int(line.split()[1])
        except OSError:
            continue
    return tot / 2**20


def pending_gib(jobs):
    return sum(max(0.0, j['mem'] - group_gib(j['pid'])) for j in jobs)


def register(name, pid, mem, gpu):
    st = start_time(pid)
    if st is None:
        raise SystemExit(f'register: no process {pid}')
    with open(REG, 'a') as f:
        f.write(json.dumps(dict(name=name, pid=pid, start=st, mem=mem, gpu=gpu, t=now())) + '\n')


def admit(a):
    cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
    waiting = False
    while True:
        with open(LOCK, 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            jobs = live_jobs()
            fx, pend, gpu = fixed_gib(), pending_gib(jobs), free_gpu_mib()
            if fx + pend + a.mem <= BUDGET and gpu >= a.gpu:
                with open(a.log, 'a') as out:
                    p = subprocess.Popen(cmd, cwd=a.cwd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                         start_new_session=True, close_fds=True,
                                         preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
                tmp = a.pidfile + '.tmp'
                with open(tmp, 'w') as f:
                    f.write(f'{p.pid}\n')
                os.replace(tmp, a.pidfile)
                register(a.name, p.pid, a.mem, a.gpu)
                print(f'started {a.name} (pid {p.pid}; fixed {fx:.0f} GiB + pending {pend:.0f} + its {a.mem:.0f} '
                      f'<= {BUDGET:.0f}; GPU {gpu} MiB free, needs {a.gpu})', flush=True)
                time.sleep(RAMP if a.gpu else 0)  # the lock held: the next check sees this job's GPU memory
                return
            if not waiting:
                print(f'{a.name} waits: fixed {fx:.0f} GiB + pending {pend:.0f} + its {a.mem:.0f} vs {BUDGET:.0f}; '
                      f'GPU {gpu} MiB free, needs {a.gpu}', flush=True)
                waiting = True
        time.sleep(60)


def signal_group(j, sig):
    try:
        os.killpg(j['pid'], sig)
        return True
    except ProcessLookupError:
        return False


def watch(_):
    print(f'{now()} watch: fixed memory pause above {PAUSE} GiB, SIGTERM above {KILL}, resume below {RESUME}; '
          f'budget for starts {BUDGET}', flush=True)
    paused, last_act, last_print = [], 0.0, 0.0  # paused: registry entries, in pause order
    for j in live_jobs():
        signal_group(j, signal.SIGCONT)
    while True:
        fx, t = fixed_gib(), time.time()
        jobs = live_jobs()
        live = {(j['pid'], j['start']) for j in jobs}
        paused = [j for j in paused if (j['pid'], j['start']) in live]
        if fx > KILL and t - last_act > PAUSE_STEP and jobs:
            j = jobs[-1]
            signal_group(j, signal.SIGTERM)
            signal_group(j, signal.SIGCONT)
            print(f'{now()} fixed {fx:.1f} GiB > {KILL}: SIGTERM to {j["name"]} (pid {j["pid"]})', flush=True)
            last_act = t
        elif fx > PAUSE and t - last_act > PAUSE_STEP:
            run = [j for j in jobs if j not in paused]
            if run:
                j = run[-1]
                if signal_group(j, signal.SIGSTOP):
                    paused.append(j)
                    print(f'{now()} fixed {fx:.1f} GiB > {PAUSE}: paused {j["name"]} (pid {j["pid"]})', flush=True)
                last_act = t
        elif fx < RESUME and paused and t - last_act > PAUSE_STEP:
            j = paused.pop()
            signal_group(j, signal.SIGCONT)
            print(f'{now()} fixed {fx:.1f} GiB < {RESUME}: resumed {j["name"]} (pid {j["pid"]})', flush=True)
            last_act = t
        if t - last_print > 1800:
            print(f'{now()} fixed {fx:.1f} GiB, {len(jobs)} registered jobs running, {len(paused)} paused', flush=True)
            last_print = t
        time.sleep(POLL)


def status(_):
    jobs = live_jobs()
    print(f'fixed {fixed_gib():.1f} GiB, budget {BUDGET}, GPU {free_gpu_mib()} MiB free')
    for j in jobs:
        print(f'  {j["name"]:28s} pid {j["pid"]:7d}  uses {group_gib(j["pid"]):5.1f} GiB of {j["mem"]:4.0f}  '
              f'since {j["t"]}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='what', required=True)
    p = sub.add_parser('admit')
    p.add_argument('--name', required=True)
    p.add_argument('--mem', type=float, required=True, help='GiB of fixed memory the job takes at most')
    p.add_argument('--gpu', type=int, default=0, help='MiB of GPU memory the job needs free')
    p.add_argument('--pidfile', required=True)
    p.add_argument('--log', required=True)
    p.add_argument('--cwd', default=None)
    p.add_argument('cmd', nargs=argparse.REMAINDER)
    p = sub.add_parser('register')
    p.add_argument('--name', required=True)
    p.add_argument('--pid', type=int, required=True, help='the job\'s process group leader')
    p.add_argument('--mem', type=float, required=True)
    p.add_argument('--gpu', type=int, default=0)
    sub.add_parser('watch')
    sub.add_parser('status')
    a = ap.parse_args()
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    if a.what == 'admit':
        admit(a)
    elif a.what == 'register':
        register(a.name, a.pid, a.mem, a.gpu)
    elif a.what == 'watch':
        watch(a)
    else:
        status(a)


if __name__ == '__main__':
    sys.exit(main())
