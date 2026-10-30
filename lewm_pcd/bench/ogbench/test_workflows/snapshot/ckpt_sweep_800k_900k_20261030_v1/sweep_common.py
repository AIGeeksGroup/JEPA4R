"""Shared checkpoint selection and audit helpers for the isolated sweep."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1")
SWEEP_ROOT = ROOT / "ckpt_sweep_800k_900k_20261030_v1"
VALID_UPDATES = (800000, 900000, 1000000)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    os.replace(tmp, path)


def checkpoint_facts(
    training_config: dict,
    update: int,
    acceptance: dict,
) -> dict:
    if update not in VALID_UPDATES:
        raise ValueError(f"unsupported update {update}")
    run = Path(training_config["output"]).resolve()
    checkpoint = run / "checkpoints" / f"update_{update:09d}"
    manifest_path = checkpoint / "manifest.json"
    params_path = checkpoint / f"params_{update}.pkl"
    sidecar_path = checkpoint / "run_state.pkl"
    for path in (manifest_path, params_path, sidecar_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    expected_manifest = {
        item["counter"]: item["manifest_sha256"]
        for item in acceptance.get("checkpoint_transactions", [])
    }
    if int(manifest["counter"]) != update:
        raise RuntimeError(f"{checkpoint}: manifest counter mismatch")
    if update not in expected_manifest:
        raise RuntimeError(f"{checkpoint}: update absent from 1M acceptance ledger")

    manifest_sha = sha(manifest_path)
    if manifest_sha != expected_manifest[update]:
        raise RuntimeError(f"{checkpoint}: manifest SHA differs from acceptance ledger")
    params_sha = sha(params_path)
    sidecar_sha = sha(sidecar_path)
    if params_sha != manifest["agent_sha256"]:
        raise RuntimeError(f"{checkpoint}: params SHA mismatch")
    if sidecar_sha != manifest["sidecar_sha256"]:
        raise RuntimeError(f"{checkpoint}: run_state SHA mismatch")

    with sidecar_path.open("rb") as stream:
        sidecar = pickle.load(stream)
    if sidecar.get("network_step") != update + 1:
        raise RuntimeError(f"{checkpoint}: network_step mismatch")
    if sidecar.get("adam_count") != update:
        raise RuntimeError(f"{checkpoint}: adam_count mismatch")
    if sidecar.get("binding", {}).get("config") != training_config:
        raise RuntimeError(f"{checkpoint}: run_state training binding mismatch")

    return {
        "update": update,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "params": str(params_path),
        "params_sha256": params_sha,
        "run_state": str(sidecar_path),
        "run_state_sha256": sidecar_sha,
        "network_step": update + 1,
        "adam_count": update,
        "training_output": str(run),
        "run_state_binding_sha256": canonical_sha(sidecar["binding"]),
    }


def _upstream_identity(training: dict, group_cfg: dict) -> dict:
    encoded = {
        "schema_version": training.get("schema_version"),
        "group_id": training.get("group_id"),
        "role": training.get("role"),
        "cube_count": training.get("cube_count"),
        "seed": training.get("seed"),
        "batch_size": training.get("batch_size"),
        "input_contract": training.get("input_contract"),
        "data_binding": training.get("data_binding"),
        "cache_binding": training.get("cache_binding"),
        "feature_loader_arguments": training.get("feature_loader_arguments"),
        "encoder_binding": training.get("encoder_binding"),
        "wm": str(group_cfg.get("wm")) if group_cfg.get("wm") else None,
        "wm_sha256": group_cfg.get("wm_sha256"),
    }
    return encoded


def install_selected_loader(core: types.ModuleType, group: str, update: int) -> dict:
    """Replace only the final-1M agent loader while preserving core.main/evaluate."""
    cfg = core.GROUPS[group]
    training_path = Path(cfg["training"]).resolve()
    training = read_json(training_path)
    acceptance_path = Path(cfg["acceptance"]).resolve()
    acceptance = read_json(acceptance_path)
    if acceptance.get("actual_exit_code") != 0:
        raise RuntimeError(f"{group}: source training acceptance is not successful")
    if acceptance.get("actual_current_dim") != cfg["current_dim"]:
        raise RuntimeError(f"{group}: acceptance current-dim mismatch")
    if acceptance.get("actual_goal_dim") != cfg["goal_dim"]:
        raise RuntimeError(f"{group}: acceptance goal-dim mismatch")

    facts = checkpoint_facts(training, update, acceptance)
    cfg["final"] = Path(facts["checkpoint"])
    cfg["manifest_sha256"] = facts["manifest_sha256"]

    def selected_load_agent(_group: str):
        if _group != group:
            raise RuntimeError(f"selected loader bound to {group}, got {_group}")
        sys.path[:0] = [
            str(core.CASE),
            str(core.ROOT),
            str(core.ROOT / "source/ogbench/impls"),
            str(core.ROOT / "source/ogbench"),
        ]
        import gciql_boundary as boundary

        directory, sidecar = boundary.restore_saved(
            training, Path(training["output"]), update
        )
        if Path(directory).resolve() != Path(facts["checkpoint"]).resolve():
            raise RuntimeError("restore_saved selected a different checkpoint")
        if sidecar["network_step"] != update + 1 or sidecar["adam_count"] != update:
            raise RuntimeError("selected sidecar counters mismatch")

        gate_path = Path(cfg["gate"])
        with np.load(gate_path, allow_pickle=False) as saved:
            example = {
                key: saved[key][:1].copy()
                for key in ("observations", "actions", "actor_goals")
            }
        template = boundary.create_agent(
            boundary.model.GCIQLAgent,
            int(training.get("seed", 0)),
            example,
            boundary.agent_config(training),
        )
        agent = boundary.restore_agent(template, str(directory), update)
        if boundary.jax.default_backend() != "cpu":
            raise RuntimeError("selected GCIQL restore did not run on CPU")
        leaves = boundary.jax.tree_util.tree_leaves(agent)
        if len(leaves) != 153:
            raise RuntimeError(f"selected agent leaf count {len(leaves)} != 153")
        if any(not np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
            raise RuntimeError("selected agent contains non-finite leaves")
        if int(agent.network.step) != update + 1:
            raise RuntimeError("restored agent network_step mismatch")
        if int(agent.network.opt_state[0].count) != update:
            raise RuntimeError("restored agent Adam count mismatch")
        fingerprint = boundary.fingerprint(agent)
        if int(fingerprint["leaves"]) != 153:
            raise RuntimeError("selected agent fingerprint leaf count mismatch")
        if update == 1000000 and fingerprint != acceptance.get("state"):
            raise RuntimeError("1M regression fingerprint differs from acceptance")
        facts["agent_fingerprint"] = fingerprint
        facts["jax_backend"] = boundary.jax.default_backend()
        facts["agent_leaf_count"] = len(leaves)
        return agent

    upstream = _upstream_identity(training, cfg)
    binding = {
        "actual_exit_code": 0,
        "schema_version": 1,
        "group_id": group,
        "selected_update": update,
        "checkpoint": facts,
        "training_config": str(training_path),
        "training_config_sha256": sha(training_path),
        "source_acceptance": str(acceptance_path),
        "source_acceptance_sha256": sha(acceptance_path),
        "gate": str(Path(cfg["gate"]).resolve()),
        "gate_sha256": sha(Path(cfg["gate"]).resolve()),
        "source_acceptance_ledger_manifest_sha256": {
            item["counter"]: item["manifest_sha256"]
            for item in acceptance["checkpoint_transactions"]
        },
        "upstream_identity": upstream,
        "upstream_identity_sha256": canonical_sha(upstream),
        "restore_contract": {
            "checkpoint_counter": update,
            "adam_count": update,
            "network_step": update + 1,
            "strict_agent_restore": True,
            "cpu_jax": True,
            "agent_leaf_count": 153,
            "all_leaves_finite": True,
            "one_m_state_exact": update == 1000000,
        },
    }
    core.load_agent = selected_load_agent
    return binding


def independent_cpu_reload(training_path: Path, gate_path: Path, update: int) -> dict:
    """Reload the selected policy in a fresh audit process and return its fingerprint."""
    training_path = Path(training_path).resolve()
    gate_path = Path(gate_path).resolve()
    training = read_json(training_path)
    sys.path[:0] = [
        str(ROOT / "cube23_distance_gciql_v1"),
        str(ROOT),
        str(ROOT / "source/ogbench/impls"),
        str(ROOT / "source/ogbench"),
    ]
    import gciql_boundary as boundary

    directory, sidecar = boundary.restore_saved(
        training, Path(training["output"]), update
    )
    with np.load(gate_path, allow_pickle=False) as saved:
        example = {
            key: saved[key][:1].copy()
            for key in ("observations", "actions", "actor_goals")
        }
    template = boundary.create_agent(
        boundary.model.GCIQLAgent,
        int(training.get("seed", 0)),
        example,
        boundary.agent_config(training),
    )
    agent = boundary.restore_agent(template, str(directory), update)
    leaves = boundary.jax.tree_util.tree_leaves(agent)
    if boundary.jax.default_backend() != "cpu":
        raise RuntimeError("independent reload backend is not CPU")
    if len(leaves) != 153:
        raise RuntimeError("independent reload leaf count mismatch")
    if any(not np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise RuntimeError("independent reload contains non-finite leaves")
    if int(agent.network.step) != update + 1:
        raise RuntimeError("independent reload network_step mismatch")
    if int(agent.network.opt_state[0].count) != update:
        raise RuntimeError("independent reload Adam count mismatch")
    if sidecar["network_step"] != update + 1 or sidecar["adam_count"] != update:
        raise RuntimeError("independent reload sidecar mismatch")
    return {
        "actual_exit_code": 0,
        "backend": boundary.jax.default_backend(),
        "checkpoint": str(directory),
        "network_step": int(agent.network.step),
        "adam_count": int(agent.network.opt_state[0].count),
        "agent_leaf_count": len(leaves),
        "all_leaves_finite": True,
        "agent_fingerprint": boundary.fingerprint(agent),
        "training_config_sha256": sha(training_path),
        "gate_sha256": sha(gate_path),
    }


def verify_output(output: Path, group: str, mode: str, current_dim: int, goal_dim: int, max_steps: int) -> dict:
    """Reuse the preserved independent reporter and additionally read step arrays."""
    output = Path(output)
    audit_path = ROOT / "cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/audit.py"
    expected_audit_sha = "ffccacb2f94cfd37f0480419642341e7c966e490e1a97e6056eecfe0033c5285"
    if sha(audit_path) != expected_audit_sha:
        raise RuntimeError("preserved result auditor SHA changed")
    source = audit_path.read_text()
    if source.count("211") != 3:
        raise RuntimeError("preserved result auditor width substitution contract changed")
    module = types.ModuleType("_checkpoint_sweep_result_auditor")
    exec(
        compile(source.replace("211", str(current_dim)), str(audit_path), "exec"),
        module.__dict__,
    )
    expected = 1 if mode == "tiny" else 250
    cap = 4 if mode == "tiny" else max_steps
    result = module.verify_report(
        SWEEP_ROOT,
        group,
        "diagnostic" if mode == "tiny" else "full",
        output,
        expected,
        goal_dim,
        cap,
    )
    episodes = sorted((output / "episodes").glob("task*/episode_*"))
    if len(episodes) != expected:
        raise RuntimeError("episode directory count mismatch")
    for episode in episodes:
        item = read_json(episode / "result.json")
        with np.load(episode / "steps.npz", allow_pickle=False) as data:
            actions = np.asarray(data["actions"])
            rewards = np.asarray(data["rewards"])
            successes = np.asarray(data["successes"])
        if actions.shape != (item["steps"], 5):
            raise RuntimeError(f"{episode}: action shape mismatch")
        if rewards.shape != (item["steps"],) or successes.shape != (item["steps"],):
            raise RuntimeError(f"{episode}: reward/success shape mismatch")
        if not np.isfinite(actions).all() or not np.isfinite(rewards).all():
            raise RuntimeError(f"{episode}: non-finite step arrays")
        if np.abs(actions).max() > 1.000001:
            raise RuntimeError(f"{episode}: action outside [-1,1]")
        if int(successes[-1]) != int(item["success"]):
            raise RuntimeError(f"{episode}: final success mismatch")
        rows = item.get("policy_input_contracts", [])
        if len(rows) != item["steps"]:
            raise RuntimeError(f"{episode}: policy-input row count mismatch")
        if any(
            row.get("current_dim") != current_dim
            or row.get("goal_dim") != goal_dim
            or row.get("current_finite") is not True
            or row.get("goal_finite") is not True
            for row in rows
        ):
            raise RuntimeError(f"{episode}: policy-input contract mismatch")
        if group in {"G06", "G09", "G12"}:
            proof = item.get("goal_capture_proof", {})
            if (
                proof.get("physics_unchanged") is not True
                or proof.get("env_rng_unchanged") is not True
                or proof.get("global_rng_unchanged") is not True
                or proof.get("geometry_offsamples") != 0
                or item.get("camera", {}).get("geometry_offsamples") != 0
            ):
                raise RuntimeError(f"{episode}: PCD goal-capture invariant failed")
    result.update(
        actual_exit_code=0,
        mode=mode,
        group_id=group,
        current_dim=current_dim,
        goal_dim=goal_dim,
        max_episode_steps=cap,
        actual_step_arrays_checked=True,
        original_result_auditor_sha256=expected_audit_sha,
        no_NN_or_update_performed=True,
    )
    return result
