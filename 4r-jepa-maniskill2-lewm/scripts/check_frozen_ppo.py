#!/usr/bin/env python
"""Run bounded CPU-only stage-three checks and save auditable test evidence.

Source WM weights, observations, actions, rewards and episode flags are SYNTHETIC
compatibility fixtures. This does not collect data, create an environment or
renderer, start PPO training in a simulator, or establish representation quality.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
def main():
    started = time.monotonic()
    output = ROOT / "artifacts/tests" / datetime.now(timezone.utc).strftime(
        "frozen_ppo_cpu_%Y%m%dT%H%M%S_%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    fixture = output / "SYNTHETIC_worldmodel" / "weights_epoch_1.pt"
    environ = os.environ.copy()
    environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        environ[name] = "1"
    environ["PYTHONDONTWRITEBYTECODE"] = "1"
    environ["MS2_LEWM_TEST_WM_FIXTURE"] = str(fixture)
    paths = [ROOT / "src", ROOT / "vendor/ManiSkill2-Learn", ROOT / "vendor/pytorch3d",
             ROOT / "vendor/le-wm", ROOT / "vendor/stable-worldmodel-lewm-compatible"]
    environ["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    if hasattr(os, "nice") and os.nice(0) < 19:
        os.nice(19 - os.nice(0))
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_frozen_ppo.py",
               "--junitxml", str(output / "pytest.xml"), "--basetemp", str(output / "pytest-tmp")]
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environ, stdout=log,
                                stderr=subprocess.STDOUT, timeout=180)
    counts = {}
    if (output / "pytest.xml").is_file():
        suites = ET.parse(output / "pytest.xml").getroot().iter("testsuite")
        for suite in suites:
            for key in ("tests", "failures", "errors", "skipped"):
                counts[key] = counts.get(key, 0) + int(suite.get(key, 0))
    passed = result.returncode == 0 and counts.get("tests", 0) > 0
    report = dict(
        passed=passed,
        notice="CPU SYNTHETIC compatibility fixtures only; no expert/policy-quality evidence",
        fixture=str(fixture),
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest() if fixture.is_file() else None,
        production_config=str(ROOT / "configs/ppo_frozen_wm.py"),
        synthetic_fields=["worldmodel_training_data", "obs", "next_obs", "actions", "rewards", "dones", "episode_dones"],
        simulator_created=False,
        renderer_created=False,
        cuda_initialized=False if passed else "See test log; passing tests assert CUDA is uninitialized",
        checks=counts,
        checked_contracts=[
            "Exact WM encoder and temporal weights loaded through latent2; predictor and LeWMLoss omitted",
            "Frozen encoder stays eval and deterministic under agent.train and shuffled observations",
            "Both optimizers exclude encoder; original actor/critic take latent2 plus current state (512+30=542)",
            "Actual upstream PPO update changes actor, critic and Gaussian log_std with finite gradients",
            "Encoder parameters and buffers remain byte-identical after PPO updates",
            "Strict upstream PPO checkpoint and optimizer restore; identical subsequent optimization step",
            "WM checkpoint SHA-256 enforced",
            "Consecutive training/control frames: stride 1; legacy stride-5 world-model checkpoint rejected",
            "Encoder bundle loads independently of original WM; changed configuration or implementation hashes rejected",
        ],
        command=command,
        elapsed_seconds=time.monotonic() - started,
        log=str(output / "pytest.log"),
        junit=str(output / "pytest.xml"),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        print((output / "pytest.log").read_text()[-18000:], flush=True)
    return result.returncode if result.returncode else (0 if passed else 1)


if __name__ == "__main__":
    sys.exit(main())
