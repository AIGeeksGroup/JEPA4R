"""Command line. Installed entry points: ``jepa4r-run``, ``jepa4r-reproduce``, ``jepa4r-report``;
without installation use ``python -m jepa4r <command>``."""

import importlib
import sys
from datetime import datetime, timezone

from omegaconf import OmegaConf

from .config import REPO_ROOT, RUN_KEYS, STAGES, available, compose, run_dir, run_key
from .utils.io import write_metrics

STAGE_USAGE = """\
usage: jepa4r-run <stage> [key=value ...]

key=value selects configs (env / method / policy / decoder) or overrides a value (a.b.c=value):
  collect_wm_data      env=E
  prepare_policy_data  env=E
  train_wm             env=E method=M [tag=T]
  train_policy         env=E method=M [tag=T] [policy=P]
  train_decoder        env=E decoder=D [tag=T]
  eval_recon           env=E decoder=D [tag=T]
  verify_twin          env=E
  count_params         method=M | decoder=D
"""

USAGE = """\
usage: python -m jepa4r <command> [args]

  <stage> [key=value ...]   one run of a pipeline stage           (= jepa4r-run)
  reproduce SPEC [...]      jobs of experiment specs              (= jepa4r-reproduce)
  report                    paper vs. reproduced numbers          (= jepa4r-report)
  config [key=value ...]    print a composed config
  list                      config names and experiment specs
"""


def run_stage(stage: str, args: list[str]) -> int:
    cfg = compose(args)
    present = [k for k in RUN_KEYS[stage] if k in cfg]
    if stage == "count_params":
        if len(present) != 1:
            raise SystemExit("count_params needs exactly one of method=... / decoder=...")
    elif len(present) != len(RUN_KEYS[stage]):
        raise SystemExit(f"{stage} needs " + " ".join(f"{k}=..." for k in RUN_KEYS[stage]))
    out_dir = run_dir(cfg, stage)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    module = importlib.import_module(f"jepa4r.stages.{stage}")
    metrics = module.run(cfg, out_dir)
    if metrics is not None:
        write_metrics(out_dir, run_key(cfg, stage), metrics)
    (out_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(f"[jepa4r] {stage} finished: {out_dir}")
    return 0


def run_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(STAGE_USAGE)
        return 0
    stage = argv[0].replace("-", "_")
    if stage not in STAGES:
        print(f"unknown stage {argv[0]!r}\n\n{STAGE_USAGE}")
        return 2
    return run_stage(stage, argv[1:])


def list_configs() -> int:
    for group in ("env", "method", "policy", "decoder", "encoder"):
        print(f"{group:<8} " + " ".join(available(group)))
    specs = sorted((REPO_ROOT / "experiments").glob("*.yaml"))
    print("specs    " + " ".join(f"experiments/{p.name}" for p in specs))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    command, rest = argv[0].replace("-", "_"), argv[1:]
    if command in STAGES:
        return run_stage(command, rest)
    if command == "reproduce":
        from .orchestrate import main as reproduce

        return reproduce(rest)
    if command == "report":
        from .report import main as report

        return report(rest)
    if command == "config":
        print(OmegaConf.to_yaml(compose(rest)))
        return 0
    if command == "list":
        return list_configs()
    print(f"unknown command {argv[0]!r}\n\n{USAGE}")
    return 2
