#!/usr/bin/env bash
# Evaluate every new per-epoch checkpoint of the point-cloud runs with the official eval pipeline
# (config/eval/cube_pcd.yaml: 50 starts, seed 42, CEM, budget 50), as they appear -- WITHOUT ever
# competing with training for memory:
#   * one eval at a time, `nice -n 10`
#   * GPU only if >= 12 GB VRAM is free; otherwise the whole eval runs on CPU (osmesa rendering,
#     device=cpu solver) so training's CUDA allocations are never touched
#   * launch only while cgroup anon memory < 40 GB; while running, kill the eval if anon exceeds 60 GB
#     or the eval's own RSS exceeds 30 GB (retried later, max 3 tries). memory.current is not used: the
#     datasets' page cache fills it to the 90 GB cap and is reclaimable; anon is what can OOM.
# Run detached:  setsid nohup bash scripts/eval/auto_eval_checkpoints.sh > $STABLEWM_HOME/logs/eval/auto_eval.log 2>&1 < /dev/null &
# Results: $STABLEWM_HOME/logs/eval/success_rates.csv  (run,epoch,success_rate,eval_seconds,backend,finished_at)
set -uo pipefail
P=/root/autodl-tmp/lewm234d/lewm_pcd
source "$P/env.sh"
PY=/root/autodl-tmp/conda/envs/lewm_pcd/bin/python
RUNS=${RUNS:-"cube_pcd_ms2 cube_pcd_dexart cube_pcd_ppt cube_pcd_agg_ppt cube_pcd_agg_dexart cube_pcd_agg_ms2"}
EXTRA=${EXTRA:-}                      # extra hydra overrides (tests)
# memory.current includes reclaimable page cache (the 37 GB dataset lives there); only anon is non-reclaimable
LAUNCH_MAX_GB=${LAUNCH_MAX_GB:-82}; LAUNCH_MAX_ANON_GB=${LAUNCH_MAX_ANON_GB:-40}
KILL_GB=${KILL_GB:-87}; KILL_ANON_GB=${KILL_ANON_GB:-60}; KILL_RSS_GB=${KILL_RSS_GB:-30}; GPU_MIN_FREE_MB=${GPU_MIN_FREE_MB:-12000}
OUT=$STABLEWM_HOME/logs/eval; mkdir -p "$OUT/auto"; CSV=${CSV:-$OUT/success_rates.csv}; SUFFIX=${SUFFIX:-}   # CSV/SUFFIX: alternative protocol runs (e.g. EXTRA="solver.n_steps=10" CSV=.../success_rates_cem10.csv SUFFIX=_cem10)
[ -f "$CSV" ] || echo "run,epoch,success_rate,eval_seconds,backend,finished_at" > "$CSV"
cd "$P"
cg_gb()   { local v=$(cat /sys/fs/cgroup/memory.current 2>/dev/null); echo $(( ${v:-0} / 1073741824 )); }
anon_gb() { local v=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat 2>/dev/null | awk '{print $2}'); echo $(( ${v:-0} / 1073741824 )); }
gpu_free_mb() { local t=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1); local u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); echo $(( ${t:-0} - ${u:-0} )); }
log() { echo "$(date +%F_%T) $*"; }

eval_one() {  # run epoch
  local r=$1 e=$2 tag=${1}_epoch${2}${SUFFIX} log=$OUT/auto/${1}_epoch${2}${SUFFIX}.log backend dev cfgname=cube_pcd.yaml extra=""
  case "$r" in
    *pix_agg*) cfgname=cube_pix_agg.yaml;;   # 4D pixel: history_len 3 + goal window 3 on the official pixel protocol
    *pix*)     cfgname=cube_pix.yaml;;       # pixel baseline: the official cube.yaml + device/video knobs
    *agg*)     cfgname=cube_pcd_agg.yaml;;   # 4D point cloud
  esac
  if [ "${GPU_ONLY:-0}" = 1 ]; then while [ "$(gpu_free_mb)" -lt "$GPU_MIN_FREE_MB" ]; do sleep 15; done; fi   # GPU_ONLY=1: never fall back to CPU, wait for VRAM (several loops can share the GPU)
  if [ "$(gpu_free_mb)" -ge "$GPU_MIN_FREE_MB" ]; then backend=egl; dev=cuda; else backend=osmesa; dev=cpu; fi
  local t0=$(date +%s)
  MUJOCO_GL=$backend PYOPENGL_PLATFORM=$backend LP_NUM_THREADS=8 OMP_NUM_THREADS=8 \
    nice -n 10 "$PY" eval.py --config-name=$cfgname policy="$r/weights_epoch_${e}.pt" device=$dev solver.device=$dev \
    hydra.run.dir="$OUT/auto/hydra_${tag}" $extra $EXTRA > "$log" 2>&1 &
  local pid=$! killed=""
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local rss_kb=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); local rss=$(( ${rss_kb:-0} / 1048576 ))
    if [ "$(anon_gb)" -ge "$KILL_ANON_GB" ] || [ "$rss" -ge "$KILL_RSS_GB" ]; then
      log "MEMORY GUARD: killing eval $tag (cgroup $(cg_gb) GB, anon $(anon_gb) GB, eval rss ${rss} GB)"; kill $pid; sleep 3; kill -9 $pid 2>/dev/null; killed=1; break
    fi
  done
  wait $pid 2>/dev/null; local ec=$?
  local sr=$(grep -oE "'success_rate': [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -n "$killed" ]; then
    echo x >> "$OUT/auto/${tag}.attempts"; mv "$log" "$log.killed$(wc -l < "$OUT/auto/${tag}.attempts")"; return
  fi
  echo "$r,$e,${sr:-ERROR(exit $ec)},$(( $(date +%s) - t0 )),$backend/$dev,$(date +%F_%T)" >> "$CSV"
  log "done $tag -> ${sr:-ERROR} (${backend}/${dev}, $(( $(date +%s) - t0 )) s)"
}

log "auto-eval loop started (runs: $RUNS)"
while true; do
  for r in $RUNS; do
    for ck in $(ls "$STABLEWM_HOME/checkpoints/$r"/weights_epoch_*.pt 2>/dev/null | sort -V); do
      e=$(basename "$ck" .pt | grep -oE "[0-9]+$"); tag=${r}_epoch${e}${SUFFIX}
      grep -q "^$r,$e," "$CSV" && continue
      [ -f "$OUT/auto/${tag}.attempts" ] && [ "$(wc -l < "$OUT/auto/${tag}.attempts")" -ge 3 ] && continue
      while [ "$(anon_gb)" -ge "$LAUNCH_MAX_ANON_GB" ]; do   # anon only: page cache (the datasets) fills memory.current to the cap and is reclaimable
        log "waiting: cgroup $(cg_gb) GB / anon $(anon_gb) GB too high to start an eval"; sleep 300
      done
      log "evaluating $tag (cgroup $(cg_gb) GB, gpu free $(gpu_free_mb) MB)"
      eval_one "$r" "$e"          # sequential: one eval at a time
    done
  done
  sleep 120
done
