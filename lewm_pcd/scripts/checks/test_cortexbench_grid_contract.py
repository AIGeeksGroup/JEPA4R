from pathlib import Path

from lewm_pcd.bench.cortexbench.protocol import build_job_contract


def test_grid_builds_explicit_adroit_100_demo_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    contract = build_job_contract("adroit", "pen-v0", "vc1_vitb", 100, "/demos", "/runs", False)
    assert contract["num_demos"] == 100
    assert contract["eval_num_traj"] == 25
    assert contract["result_scope"] == "official RGB"
