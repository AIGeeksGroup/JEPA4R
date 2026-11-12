"""Expand experiment specs (``experiments/*.yaml``) into stage jobs and run them: ``jepa4r-reproduce``.

Spec format::

    name: tab1_ogbench
    expected: [paper/expected/tab1_ogbench.yaml]   # paper numbers this spec reproduces
    variants: {<tag>: {<dotted.key>: value}}        # optional; default {default: {}}
    jobs:
      - stage: train_wm                             # one of jepa4r.config.STAGES
        grid: {env: [...], method: [...]}           # cartesian product of group selections
        variants: {...}                             # optional; replaces the spec-level variants
        overrides: {<dotted.key>: value}            # optional; applied to every job of the block
        optional: true                              # optional; skipped unless --with-optional

Every variant becomes ``tag=<name>`` plus its overrides; data stages (collect_*) ignore variants.
Other top-level keys are free (e.g. YAML anchors). Jobs are de-duplicated, order preserved.
"""
import argparse
import itertools
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import DATA_STAGES, GROUPS, REPO_ROOT, STAGES, compose, run_dir

BLOCK_KEYS = {"stage", "grid", "variants", "overrides", "optional"}


@dataclass(frozen=True)
class Job:
    stage: str
    selections: tuple          # ((group, name), ..., ("tag", tag))
    overrides: tuple = ()      # ((dotted.key, value), ...)
    optional: bool = False

    def args(self) -> list[str]:
        return [f"{k}={v}" for k, v in self.selections] + [f"{k}={_fmt(v)}" for k, v in self.overrides]

    def argv(self, python: str = sys.executable) -> list[str]:
        return [python, "-m", "jepa4r", self.stage, *self.args()]

    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in ["python", "-m", "jepa4r", self.stage, *self.args()])

    @property
    def keys(self) -> dict:
        return dict(self.selections)


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return yaml.safe_dump(value, default_flow_style=True).strip()
    return str(value)


def load_spec(path) -> dict:
    path = Path(path)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    for i, block in enumerate(spec.get("jobs") or []):
        where = f"{path.name}: jobs[{i}]"
        if block.get("stage") not in STAGES:
            raise ValueError(f"{where}: unknown stage {block.get('stage')!r}")
        if set(block) - BLOCK_KEYS:
            raise ValueError(f"{where}: unknown keys {sorted(set(block) - BLOCK_KEYS)}")
        if set(block.get("grid") or {}) - set(GROUPS):
            raise ValueError(f"{where}: grid keys must be among {GROUPS}")
    spec.setdefault("_path", str(path))
    return spec


def expand(spec: dict, *, with_optional: bool = False, stages=None, where: dict | None = None) -> list[Job]:
    """All jobs of a spec, filtered by stage and by ``where`` (e.g. {"env": "scene"})."""
    spec_variants = spec.get("variants") or {"default": {}}
    jobs = []
    for block in spec["jobs"]:
        stage, optional = block["stage"], bool(block.get("optional", False))
        if (optional and not with_optional) or (stages and stage not in stages):
            continue
        grid = block.get("grid") or {}
        variants = {None: {}} if stage in DATA_STAGES else (block.get("variants") or spec_variants)
        base_overrides = block.get("overrides") or {}
        for combo in itertools.product(*grid.values()):
            selection = dict(zip(grid, combo))
            if where and any(k in selection and str(selection[k]) != str(v) for k, v in where.items()):
                continue
            for tag, overrides in variants.items():
                sel = tuple(selection.items()) + ((("tag", tag),) if tag is not None else ())
                ov = tuple({**base_overrides, **(overrides or {})}.items())
                jobs.append(Job(stage, sel, ov, optional))
    return dedupe(jobs)


def dedupe(jobs: list[Job]) -> list[Job]:
    seen, out = set(), []
    for job in jobs:
        ident = (job.stage, job.selections, job.overrides)
        if ident not in seen:
            seen.add(ident)
            out.append(job)
    return out


def job_dir(job: Job) -> Path:
    return run_dir(compose(job.args()), job.stage)


def is_done(job: Job) -> bool:
    return (job_dir(job) / "DONE").exists()


def run_jobs(jobs: list[Job], *, execute: bool = False, force: bool = False,
             python: str = sys.executable) -> int:
    """Print every job with its status; with ``execute`` run the pending ones sequentially."""
    n_done = 0
    for i, job in enumerate(jobs, 1):
        done = is_done(job)
        n_done += done
        status = "done" if done else ("run" if execute else "todo")
        print(f"[{i:>3}/{len(jobs)}] {status:<4} {job.command()}")
        if execute and (force or not done):
            subprocess.run(job.argv(python), check=True, cwd=REPO_ROOT)
    print(f"{n_done}/{len(jobs)} jobs done" + ("" if execute else " (dry run; add --run to execute)"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jepa4r-reproduce", description="Run the jobs of experiment specs.")
    parser.add_argument("specs", nargs="+", help="experiments/*.yaml")
    parser.add_argument("--run", action="store_true", help="execute pending jobs (default: list them)")
    parser.add_argument("--force", action="store_true", help="also re-run finished jobs")
    parser.add_argument("--with-optional", action="store_true", help="include blocks marked optional")
    parser.add_argument("--stage", action="append", choices=STAGES, help="only these stages")
    parser.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                        help="only jobs whose selection matches, e.g. --where env=scene")
    args = parser.parse_args(argv)
    where = dict(item.split("=", 1) for item in args.where)
    jobs = []
    for spec in args.specs:
        jobs += expand(load_spec(spec), with_optional=args.with_optional, stages=args.stage, where=where)
    return run_jobs(dedupe(jobs), execute=args.run, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
