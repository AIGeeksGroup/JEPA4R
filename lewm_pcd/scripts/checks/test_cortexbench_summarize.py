import json
import os
import subprocess
import sys
from pathlib import Path


def test_summarize_uses_best_of_twenty_as_primary_and_reports_final_as_secondary(tmp_path):
    runs = tmp_path / "logs" / "cortexbench" / "runs" / "lewm_adroit_rgb" / "pen-v0"
    for seed, best, final in ((100, 60.0, 10.0), (200, 70.0, 20.0), (300, 80.0, 30.0)):
        run = runs / f"s{seed}"
        run.mkdir(parents=True)
        (run / "DONE").write_text(json.dumps({
            "model": "lewm_adroit_rgb",
            "env": "pen-v0",
            "seed": seed,
            "highest_success": best,
            "final_success": final,
        }) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["STABLEWM_HOME"] = str(tmp_path)
    script = Path(__file__).resolve().parents[2] / "bench" / "cortexbench" / "summarize.py"
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    score = json.loads((tmp_path / "logs" / "cortexbench" / "results" / "scores.json").read_text())["lewm_adroit_rgb"]["pen"]
    assert score["best"] == 70.0
    assert score["best_std"] == 8.2
    assert score["last"] == 20.0
    assert score["last_std"] == 8.2
    assert score["seeds"] == {"100": 60.0, "200": 70.0, "300": 80.0}
    assert score["final_seeds"] == {"100": 10.0, "200": 20.0, "300": 30.0}

    report = (tmp_path / "logs" / "cortexbench" / "results" / "scores.md").read_text(encoding="utf-8")
    assert "Best-of-20" in report
    assert "Final" in report
