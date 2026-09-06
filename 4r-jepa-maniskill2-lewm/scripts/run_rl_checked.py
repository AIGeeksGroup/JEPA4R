"""Record resource use of our own bounded smoke child; never signal other jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pynvml

root = Path(__file__).resolve().parents[1]
mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
if mode not in {"smoke", "eval-smoke"}:
    raise SystemExit("Only bounded smoke/eval-smoke are accepted")
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)
if mem.free < 1024**3:
    raise SystemExit("Less than 1 GiB GPU memory free; no renderer started")
out = root / "artifacts"
out.mkdir(exist_ok=True)
report = dict(mode=mode, started_utc=datetime.now(timezone.utc).isoformat(),
              gpu_used_before_bytes=mem.used, gpu_free_before_bytes=mem.free,
              own_pids=[], own_peak_rss_bytes=0, own_peak_gpu_bytes=0)
with (out / f"rl_identity_{mode}.log").open("w") as log:
    proc = subprocess.Popen(["bash", str(root / "scripts/run_rl.sh"), mode], cwd=root, stdout=log, stderr=subprocess.STDOUT)
    report["launcher_pid"] = proc.pid
    print(f"Started own {mode} PID={proc.pid}, log={log.name}", flush=True)
    while proc.poll() is None:
        try:
            family = [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)
            ids = {p.pid for p in family}
            report["own_pids"] = sorted(set(report["own_pids"]) | ids)
            report["own_peak_rss_bytes"] = max(report["own_peak_rss_bytes"], sum(p.memory_info().rss for p in family))
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
            report["own_peak_gpu_bytes"] = max(report["own_peak_gpu_bytes"], sum(p.usedGpuMemory for p in processes if p.pid in ids))
        except psutil.NoSuchProcess:
            pass
        time.sleep(1)
report.update(returncode=proc.returncode, finished_utc=datetime.now(timezone.utc).isoformat(),
              gpu_used_after_bytes=pynvml.nvmlDeviceGetMemoryInfo(gpu).used)
(out / f"rl_identity_{mode}_resources.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
sys.exit(proc.returncode)
