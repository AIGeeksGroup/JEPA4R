from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ASSET_DIR = PROJECT_ROOT / "envs" / "descriptions" / "franka_emika_panda"
PANDA_XML = ASSET_DIR / "panda.xml"
PINNED_COMMIT = "8161bba264d7fa7c99ca301e91e7fb44737676ad"


def _manifest_entries() -> list[tuple[str, Path]]:
    manifest = ASSET_DIR / "SHA256SUMS"
    assert manifest.is_file(), f"missing asset manifest: {manifest}"
    entries: list[tuple[str, Path]] = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        digest, relative_path = raw_line.split(maxsplit=1)
        entries.append((digest, REPOSITORY_ROOT / relative_path.strip().lstrip("*")))
    return entries


def test_panda_assets_record_official_source_and_license() -> None:
    commit_file = ASSET_DIR / "UPSTREAM_COMMIT"
    license_file = ASSET_DIR / "LICENSE"
    assert commit_file.is_file(), f"missing provenance file: {commit_file}"
    assert commit_file.read_text(encoding="utf-8").strip() == PINNED_COMMIT
    assert license_file.is_file(), f"missing upstream license: {license_file}"
    license_text = license_file.read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_panda_asset_manifest_matches_every_vendored_file() -> None:
    entries = _manifest_entries()
    assert entries, "asset manifest must not be empty"
    manifested = set()
    for expected_digest, path in entries:
        assert path.is_file(), f"manifest path does not exist: {path}"
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual_digest == expected_digest, f"hash mismatch: {path}"
        manifested.add(path.resolve())

    expected = {
        path.resolve()
        for path in ASSET_DIR.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert manifested == expected


def test_panda_xml_compiles_with_expected_robot_identity() -> None:
    assert PANDA_XML.is_file(), f"missing official Panda model: {PANDA_XML}"
    model = mujoco.MjModel.from_xml_path(str(PANDA_XML))
    joint_names = {model.joint(index).name for index in range(model.njnt)}
    expected_joints = {f"joint{index}" for index in range(1, 8)}
    expected_joints |= {"finger_joint1", "finger_joint2"}
    assert expected_joints <= joint_names
    assert model.nu == 8
