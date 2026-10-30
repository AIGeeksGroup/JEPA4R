"""Materialize the independent Cube2 re-audit case from the accepted Cube3 source.

The script performs counted, reversible source substitutions.  It never edits
the Cube3 source or any existing Cube2 result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


REMOTE_ROOT = "/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1"
CASE_NAME = "pldm_cube2_reaudit_20261029_v1"
GROUP = "P2_cube2_pldm_reaudit_rgb"

CUBE2_ASSET_ID = "cube2_pldm_impala512_epoch10_20261021_v1"
CUBE2_ASSET = f"{REMOTE_ROOT}/assets/baselines/cube2/{CUBE2_ASSET_ID}"
CUBE2_WEIGHT_SHA = "ada09ccd9508434f688455622491b5910f2ad2190189a5f67f60cac53d7a5ae5"
CUBE2_MANIFEST_SHA = "1b7737dc8433cfe0f095e298ef544b245759135954b5c17ab369321cc1969dba"
CUBE2_ADDENDUM_SHA = "a9636533baa2000a5792911513950817464ea2343f763a9bb5d0acf6e6e11a51"
CUBE2_ORIGINAL_PLAN_SHA = "12454be9c3411d444840d323e75a7db8a3a63de79e0d231e02bfb35f29bffff9"
CUBE2_ORIGINAL_ARRAYS_SHA = "28eedab709578fdf874d2261984c7cfc26cd2b7c41b4d191fc3fc2dfd83bb35e"
CUBE2_DATA_PRODUCER_SHA = "fa6ff3dcb2d8fb709de069ea3aebfa5970cf3686a4a8aec20def899bd9848cd8"
CUBE2_DATA_AUDIT_SHA = "75eb5137c30cac3628f0e781c7d2eb4a96b29eb7689a7fcae6ce1509dc303adc"
CUBE2_RENDER_SHA = "7607611907523a384d567f093c7c76d2438c4de9968e7ce9d3ef6c883300b144"
G05_CONFIG_SHA = "8c3af1e965e3266965cfac9a1baedbfb081da1c5a15549de1574b3cdfbf8b65b"

SOURCE_HASHES = {
    "cache_v1/encoder_adapter.py": "c5b1709817a88a0ed439a4a27f36500aeca67e4e3586c6c68cc24dd03fb7cc81",
    "cache_v1/producer_bridge.py": "f012b4579894a2946884d8de99c7da3c4a56b33b80157f51f79c9d39dce12660",
    "cache_v1/audit_bridge.py": "d12b9bb010c692f6d8825c1faaef8c9d47ecacf3cf988d98463682cbb4c53fec",
    "cache_v1/prepare_plan.py": "b57d98e9c307b0e6e96ffa68afd008bed21de39d23bec697c6e3b8c105b4ddf9",
    "functional_v1/pldm_inputs.py": "45229484956cbd467b498b57146057b6f3651f283a6c900f93d27b254923ce01",
    "functional_v1/pldm_runtime.py": "9572d0819a44e91012a4c7932ecb1cb5095d2ab301b1cdd30e0e62f56f20779e",
    "functional_v1/native_sr.py": "c6ec01222f74f91d4e0263bda153c2b83d65d356ccc59f2ab08887336d4dd5f4",
    "functional_v1/audit_native.py": "379598b36ce3cdee007c9f747e01edb47f73dc8579756c34452a21df80f2b9bf",
    "functional_v1/prepare.py": "d4a1415be31368ea0bfd6d4910eaae462cc834e8dac496be22b8477e483f037c",
    "functional_v1/SOURCE_FREEZE_v2.json": "16d7028ca0a2f26a1b29299459bbbf383476480da47a7299dcdb02b52f791ba9",
    "functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE_v2.json": "e7cbc9e2f32f710e8be81086333279fdea81b409b8e46ddccb5f44c9b91c7b10",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_exact(text: str, before: str, after: str, count: int = 1) -> str:
    actual = text.count(before)
    if actual != count:
        raise ValueError(f"replacement count {actual} != {count}: {before!r}")
    return text.replace(before, after)


def source(source_root: Path, relative: str) -> bytes:
    raw = (source_root / relative).read_bytes()
    assert sha(raw) == SOURCE_HASHES[relative], relative
    return raw


def text_source(source_root: Path, relative: str) -> str:
    return source(source_root, relative).decode("utf-8")


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    assert path.read_bytes() == raw


def write_json_exclusive(path: Path, value: object) -> None:
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    write_exclusive(path, raw)


def data_producer(raw: bytes) -> bytes:
    """Reproduce the already accepted Double-specific base producer."""
    text = raw.decode("utf-8")
    edits = [
        ("{'train': 3000, 'val': 300}", "{'train': 1000, 'val': 100}", 1),
        ("3303300", "1101100", 1),
        ("completed == 3300", "completed == 1100", 1),
        ("3003000", "1001000", 1),
        ("300300", "100100", 1),
        ("('qpos', 35), ('qvel', 32)", "('qpos', 28), ('qvel', 26)", 1),
        ("qpos_width=35, qvel_width=32", "qpos_width=28, qvel_width=26", 1),
        ("shape[1] == 46", "shape[1] == 37", 1),
        ("46 * 4", "37 * 4", 2),
        ("range(3)", "range(2)", 1),
    ]
    for before, after, count in edits:
        text = replace_exact(text, before, after, count)
    result = text.encode()
    assert sha(result) == CUBE2_DATA_PRODUCER_SHA
    return result


def data_audit(raw: bytes) -> bytes:
    """Reproduce the already accepted Double-specific base cache auditor."""
    text = raw.decode("utf-8")
    edits = [
        ("{'train': 3000, 'val': 300}", "{'train': 1000, 'val': 100}", 1),
        (
            "accepted['stage'] == 'complete'",
            "accepted['data_id'] == 'cube-double-play-v0' and accepted['cube_count'] == 2",
            1,
        ),
        ("qpos_width=35, qvel_width=32", "qpos_width=28, qvel_width=26", 1),
        ("('qpos', 35), ('qvel', 32)", "('qpos', 28), ('qvel', 26)", 1),
        ("nq=35, nv=32", "nq=28, nv=26", 1),
        (
            "minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20)",
            "minimum_gain=.05, warmup_batches=3, timed_batches_per_path=20,\n"
            "            scope='original preprocessing plus frozen encode; rendering excluded; CUDA tail synchronized')",
            1,
        ),
        (
            "Require full official3000/300 cache",
            "Require full official1000/100 Double cache",
            1,
        ),
    ]
    for before, after, count in edits:
        text = replace_exact(text, before, after, count)
    result = text.encode()
    assert sha(result) == CUBE2_DATA_AUDIT_SHA
    return result


def data_render(raw: bytes) -> bytes:
    """Create the Double-specific render adapter used by the new producer."""
    text = raw.decode("utf-8")
    edits = [
        ("cube-triple-v0", "cube-double-v0", 1),
        ("(35, 32, 7)", "(28, 26, 7)", 1),
        ("range(3)", "range(2)", 2),
        ("% 3", "% 2", 1),
        ("nq=35, nv=32", "nq=28, nv=26", 1),
    ]
    for before, after, count in edits:
        text = replace_exact(text, before, after, count)
    result = text.encode()
    assert sha(result) == CUBE2_RENDER_SHA
    return result


def materialize_cache(source_root: Path, target: Path) -> list[dict]:
    records: list[dict] = []

    encoder = text_source(source_root, "cache_v1/encoder_adapter.py")
    encoder = replace_exact(encoder, "cube3_pldm_impala512_epoch10_20261021_v1", CUBE2_ASSET_ID)
    encoder = replace_exact(encoder, "96e2805301373e09cba689358e97673993af6b0a1ee4f3236b7f1ba0aa64d38a", CUBE2_MANIFEST_SHA)
    encoder = replace_exact(encoder, "c15741c0c3969c4e29d0f2cffa3820ee5dc33bb6205b895ef7375fea133f52be", CUBE2_ADDENDUM_SHA)
    encoder = replace_exact(encoder, "0ed6b328afcecf8858af3d46061bed1026a9c75ab7472a855496c907ba495b67", CUBE2_WEIGHT_SHA)
    write_exclusive(target / "cache_v1/encoder_adapter.py", encoder.encode())

    render_source_path = (
        source_root / "reference_source/official_state_2d_cache_v1/render_adapter.py"
    )
    if not render_source_path.exists():
        render_source_path = source_root.parent / "official_state_2d_cache_v1/render_adapter.py"
    render_source = render_source_path.read_bytes()
    assert sha(render_source) == "5df6dc694f7c9d38ccff69385080b1bad814f41ad6c4ae71e9def2e6ac71662f"
    write_exclusive(target / "cache_v1/reference/render_adapter.py", data_render(render_source))

    producer = text_source(source_root, "cache_v1/producer_bridge.py")
    producer = replace_exact(producer, "ADAPTER='cube3_pldm_gciql_v1/cache_v1/encoder_adapter.py'", f"ADAPTER='{CASE_NAME}/cache_v1/encoder_adapter.py'")
    producer = replace_exact(producer, "ASSET='assets/baselines/cube3/cube3_pldm_impala512_epoch10_20261021_v1'", f"ASSET='assets/baselines/cube2/{CUBE2_ASSET_ID}'")
    producer = replace_exact(
        producer,
        "def load_original():\n    text,edits=effective_source(ORIGINAL.read_bytes())\n    sys.path.insert(0,str(ORIGINAL.parent))",
        "def load_original():\n"
        "    text,edits=effective_source(ORIGINAL.read_bytes())\n"
        "    sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))",
    )
    producer = replace_exact(
        producer,
        "    text=raw.decode()\n    for before,after,count in edits:",
        "    edits.append((\"producer_hash, renderer_hash = sha(__file__), sha(HERE / 'render_adapter.py')\",\n"
        "                  \"producer_hash, renderer_hash = sha(__file__), sha(Path(__file__).with_name('reference')/'render_adapter.py')\",1))\n"
        "    text=raw.decode()\n    for before,after,count in edits:",
    )
    producer = replace_exact(
        producer,
        "    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA\n    edits=[",
        "    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA\n"
        "    data=__import__('source_overlay').data_producer(raw)\n"
        "    raw=data\n    edits=[",
    )
    write_exclusive(target / "cache_v1/producer_bridge.py", producer.encode())

    audit = text_source(source_root, "cache_v1/audit_bridge.py")
    audit = replace_exact(audit, "original.PRODUCER='cube3_pldm_gciql_v1/cache_v1/producer_bridge.py'", f"original.PRODUCER='{CASE_NAME}/cache_v1/producer_bridge.py'")
    audit = replace_exact(
        audit,
        "    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA\n    edits=[",
        "    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA\n"
        "    raw=__import__('source_overlay').data_audit(raw)\n"
        "    edits=[",
    )
    audit = replace_exact(
        audit,
        "    original.DIM=512;original.PRODUCER=",
        f"    original.DIM=512;original.ADAPTER='{CASE_NAME}/cache_v1/reference/render_adapter.py';original.PRODUCER=",
    )
    write_exclusive(target / "cache_v1/audit_bridge.py", audit.encode())

    prepare = text_source(source_root, "cache_v1/prepare_plan.py")
    prepare = replace_exact(
        prepare,
        "OLD_PLAN_SHA='e969fb9d9c00bba550173c08e4530a620950206ae2bea49e7a82fc225cc40eed'",
        f"CASE_NAME='{CASE_NAME}'\nOLD_PLAN_SHA='e969fb9d9c00bba550173c08e4530a620950206ae2bea49e7a82fc225cc40eed'",
    )
    prepare = replace_exact(prepare, "OLD_PLAN_SHA='e969fb9d9c00bba550173c08e4530a620950206ae2bea49e7a82fc225cc40eed'", f"OLD_PLAN_SHA='{CUBE2_ORIGINAL_PLAN_SHA}'")
    prepare = replace_exact(prepare, "OLD_ARRAYS_SHA='8e81a6a40662363a83be125a55d8001cc8df4e390c6fbad7277a5f7d74a09199'", f"OLD_ARRAYS_SHA='{CUBE2_ORIGINAL_ARRAYS_SHA}'")
    prepare = replace_exact(prepare, "assert case==ROOT/'cube3_pldm_gciql_v1'", f"assert case==ROOT/'{CASE_NAME}'")
    prepare = replace_exact(prepare, "original=ROOT/'official_state_2d_cache_v1/producer_plan.json'", "original=ROOT/'cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1/producer_plan.json'")
    prepare = replace_exact(prepare, "assert arrays['contract']['split_episodes']=={'train':3000,'val':300}", "assert arrays['contract']['split_episodes']=={'train':1000,'val':100}")
    prepare = replace_exact(
        prepare,
        "assert arrays['contract']['extractor_source_sha256']==ORIGINAL_SHA",
        f"assert arrays['contract']['extractor_source_sha256']=='{CUBE2_DATA_PRODUCER_SHA}'",
    )
    prepare = replace_exact(prepare, "accepted['actual_exit_code']==freeze['actual_exit_code']==0 and accepted['stage']=='complete'", "accepted['actual_exit_code']==freeze['actual_exit_code']==0 and accepted['data_id']=='cube-double-play-v0' and accepted['cube_count']==2")
    prepare = replace_exact(prepare, "for split,count in [('train',3000),('val',300)]:", "for split,count in [('train',1000),('val',100)]:")
    prepare = replace_exact(prepare, "for key,width in [('qpos',35),('qvel',32)]:", "for key,width in [('qpos',28),('qvel',26)]:")
    prepare = replace_exact(prepare, "manifest['global_step']==457030", "manifest['global_step']==378900")
    prepare = replace_exact(
        prepare,
        "    write(planpath,plan)",
        "    plan['workspace']=str(ROOT)\n    write(planpath,plan)",
    )
    prepare = replace_exact(
        prepare,
        "pins={key:value for key,value in old['source_pins'].items() if key.startswith('source/ogbench/') or key=='official_state_2d_cache_v1/render_adapter.py'}",
        "pins={key:value for key,value in old['source_pins'].items() if key.startswith('source/ogbench/') or key=='cube2_official_embeddings_v1/workspace/official_state_2d_cache_v1/render_adapter.py'}",
    )
    prepare = replace_exact(prepare, "pins['cube3_pldm_gciql_v1/cache_v1/'+name]=sha(out/name)", f"pins['{CASE_NAME}/cache_v1/'+name]=sha(out/name)")
    prepare = replace_exact(
        prepare,
        "for name in ['encoder_adapter.py','producer_bridge.py','audit_bridge.py','prepare_plan.py']:",
        "for name in ['encoder_adapter.py','producer_bridge.py','audit_bridge.py','prepare_plan.py','source_overlay.py']:",
    )
    prepare = replace_exact(prepare, "for name in ['produce.py','audit_cache.py']:pins['official_state_2d_cache_v1/'+name]=sha(ROOT/'official_state_2d_cache_v1'/name)", "for name in ['produce.py','audit_cache.py']:pins['official_state_2d_cache_v1/'+name]=sha(ROOT/'official_state_2d_cache_v1'/name)\n    pins[f'{CASE_NAME}/cache_v1/reference/render_adapter.py']=sha(out/'reference/render_adapter.py')")
    prepare = replace_exact(prepare, "encoder_family='pldm',latent_dim=512,episodes=3300,canonical_rows=3303300", "encoder_family='pldm',latent_dim=512,episodes=1100,canonical_rows=1101100")
    prepare = replace_exact(
        prepare,
        "source_arrays_original_extractor_sha256=ORIGINAL_SHA",
        f"source_arrays_original_extractor_sha256='{CUBE2_DATA_PRODUCER_SHA}'",
    )
    prepare = replace_exact(
        prepare,
        "original_extractor_sha256=ORIGINAL_SHA",
        f"original_extractor_sha256='{CUBE2_DATA_PRODUCER_SHA}'",
    )
    write_exclusive(target / "cache_v1/prepare_plan.py", prepare.encode())
    write_exclusive(
        target / "cache_v1/source_overlay.py",
        (Path(__file__).read_bytes()),
    )
    records.extend([
        {"path": "cache_v1/encoder_adapter.py", "source": "cube3 cache_v1/encoder_adapter.py"},
        {"path": "cache_v1/reference/render_adapter.py", "source": "cube3 official_state_2d_cache_v1/render_adapter.py", "effective_sha256": CUBE2_RENDER_SHA},
        {"path": "cache_v1/producer_bridge.py", "source": "cube3 cache_v1/producer_bridge.py", "data_base_sha256": CUBE2_DATA_PRODUCER_SHA},
        {"path": "cache_v1/audit_bridge.py", "source": "cube3 cache_v1/audit_bridge.py", "data_base_sha256": CUBE2_DATA_AUDIT_SHA},
        {"path": "cache_v1/prepare_plan.py", "source": "cube3 cache_v1/prepare_plan.py"},
    ])
    return records


def functional_inputs(source_root: Path) -> str:
    text = text_source(source_root, "functional_v1/pldm_inputs.py")
    text = replace_exact(text, "CHECKPOINT_SHA = '0ed6b328afcecf8858af3d46061bed1026a9c75ab7472a855496c907ba495b67'", f"CHECKPOINT_SHA = '{CUBE2_WEIGHT_SHA}'")
    text = replace_exact(text, "assert cubes == 3 and mode == 'rgb'", "assert cubes == 2 and mode == 'rgb'")
    return text


def functional_runtime(source_root: Path) -> str:
    text = text_source(source_root, "functional_v1/pldm_runtime.py")
    text = replace_exact(text, "CASE = ROOT / 'cube3_pldm_gciql_v1'", f"CASE = ROOT / '{CASE_NAME}'")
    text = replace_exact(text, "GROUP = 'P3_cube3_pldm_rgb'", f"GROUP = '{GROUP}'")
    text = replace_exact(text, "assert (config['cube_count'], config['modality']) == (3, 'rgb')", "assert (config['cube_count'], config['modality']) == (2, 'rgb')")
    text = replace_exact(text, "(531, 521)", "(531, 518)")
    text = replace_exact(text, "extension.GROUPS = {GROUP: (3, 'rgb', 531, 521)}", "extension.GROUPS = {GROUP: (2, 'rgb', 531, 518)}")
    return text


def functional_prepare(source_root: Path) -> str:
    text = text_source(source_root, "functional_v1/prepare.py")
    text = replace_exact(text, "GROUP = 'P3_cube3_pldm_rgb'", f"GROUP = '{GROUP}'")
    text = replace_exact(text, "ASSET = ROOT/'assets/baselines/cube3/cube3_pldm_impala512_epoch10_20261021_v1'", f"ASSET = ROOT/'assets/baselines/cube2/{CUBE2_ASSET_ID}'")
    text = replace_exact(text, "plan['split_episodes'] == {'train':3000, 'val':300}", "plan['split_episodes'] == {'train':1000, 'val':100}")
    text = replace_exact(text, "(3300,3303300,3300000,DIM,'float32')", "(1100,1101100,1100000,DIM,'float32')")
    text = replace_exact(text, "(3003000,300300,True)", "(1001000,100100,True)")
    text = replace_exact(text, "cfg['executable'] is False and cfg['group_id'] == GROUP", "cfg['executable'] is False and cfg['group_id'] == GROUP")
    text = replace_exact(text, "cube_count=3, data_id='cube-triple-play-v0'", "cube_count=2, data_id='cube-double-play-v0'")
    text = replace_exact(text, "split_episodes=dict(train=3000,val=300), canonical_total_rows=3303300", "split_episodes=dict(train=1000,val=100), canonical_total_rows=1101100")
    text = replace_exact(text, "current_dim=531, goal_dim=521", "current_dim=531, goal_dim=518")
    text = replace_exact(text, "cache_episodes=3300, cache_raw_rows=3303300", "cache_episodes=1100, cache_raw_rows=1101100")
    text = replace_exact(text, "(531,521,False)", "(531,518,False)")
    text = replace_exact(text, "cube_count=3,state_dim=46,current_dim=531,goal_dim=521,max_episode_steps=1000", "cube_count=2,state_dim=37,current_dim=531,goal_dim=518,max_episode_steps=500")
    text = replace_exact(text, "current_dim=531,goal_dim=521", "current_dim=531,goal_dim=518", 2)
    text = replace_exact(text, "P3_cube3_pldm_rgb.TEMPLATE.json", "P2_cube2_pldm_reaudit_rgb.TEMPLATE.json", 1)
    text = replace_exact(text, "P3_cube3_pldm_rgb.v1.json", "P2_cube2_pldm_reaudit_rgb.v1.json", 2)
    text = replace_exact(text, "cube3_pldm_gciql_v1", CASE_NAME, 1)
    text = replace_exact(text, "report_path = case/'acceptance/P3_full_acceptance.json'", f"report_path = case/'acceptance/{GROUP}_full_acceptance.json'")
    # Final recovered audit binding: the accepted 2D auditor exposes checkpoint identity via protocol.
    text = replace_exact(text, "audit['checkpoint_sha256']", "manifest['protocol']['checkpoint_sha256']")
    return text


def functional_native(source_root: Path, shim_sha: str) -> str:
    text = text_source(source_root, "functional_v1/native_sr.py")
    text = replace_exact(text, "GROUP = 'P3_cube3_pldm_rgb'", f"GROUP = '{GROUP}'")
    text = replace_exact(text, "SHIM_SHA = '9ab62959e147fcdbaff270d42102a5f7ef5c2e94724a86b08aa1bef24d8131d1'", f"SHIM_SHA = '{shim_sha}'")
    text = replace_exact(
        text,
        "shim_path = path.parent/'cube3_successor_v1/visual_sr_cube3.py'",
        "shim_path = Path(__file__).with_name('cube2_successor_v1')/'visual_sr_cube2.py'",
    )
    text = replace_exact(text, "shim.GROUPS['G08']", "shim.GROUPS['G05']")
    text = replace_exact(text, "== (3,46,'rgb',1000)", "== (2,37,'rgb',500)")
    return text


def functional_audit(source_root: Path) -> str:
    text = text_source(source_root, "functional_v1/audit_native.py")
    text = replace_exact(text, "GROUP = 'P3_cube3_pldm_rgb'", f"GROUP = '{GROUP}'")
    text = replace_exact(text, "1 if args.mode == 'tiny' else 250, 521, 4 if args.mode == 'tiny' else 1000", "1 if args.mode == 'tiny' else 250, 518, 4 if args.mode == 'tiny' else 500")
    text = replace_exact(text, "current_dim=531, goal_dim=521", "current_dim=531, goal_dim=518")
    return text


def config_template(source_root: Path) -> dict:
    cfg = json.loads(source(source_root, "functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE_v2.json"))
    cfg["role"] = GROUP
    cfg["cube_count"] = 2
    cfg["dataset_id"] = "cube-double-play-v0"
    data = cfg["data_binding"]
    data["cube_count"] = 2
    data["data_id"] = "cube-double-play-v0"
    data["env_id"] = "cube-double-v0"
    data["state_dim"] = 37
    data["object_XYZ_indices"] = [[19, 20, 21], [28, 29, 30]]
    data["goal_XYZ_dim"] = 6
    data["splits"] = {
        "train": {
            "path": f"{REMOTE_ROOT}/data/official_state/cube-double-play-v0.npz",
            "accepted_raw_sha256": "a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9",
            "accepted_stat": {"size": 297435656, "mtime_ns": 1792239630756155315, "ctime_ns": 1792239634332224443, "device": 2304, "inode": 15076767209},
            "episodes": 1000, "raw_rows": 1001000, "valid_current_rows": 1000000,
            "observation_shape": [1001000, 37], "action_shape": [1001000, 5], "dtype": "float32",
        },
        "val": {
            "path": f"{REMOTE_ROOT}/data/official_state/cube-double-play-v0-val.npz",
            "accepted_raw_sha256": "b1fcdf4bd40750351a58d0d491d6be198366ce898f0c6a2e4cb5db331966013e",
            "accepted_stat": {"size": 29726524, "mtime_ns": 1792239669888916031, "ctime_ns": 1792239670260923306, "device": 2304, "inode": 15076767205},
            "episodes": 100, "raw_rows": 100100, "valid_current_rows": 100000,
            "observation_shape": [100100, 37], "action_shape": [100100, 5], "dtype": "float32",
        },
    }
    data["data_acceptance"] = {
        "path": f"{REMOTE_ROOT}/cube2_official_embeddings_v1/data_acceptance.json",
        "sha256": "1bcafcafe815298041e42781ec8e6c25b49a9263e81d3eb1a29860e25c837cf0",
    }
    data["source_stat_freeze"] = {
        "path": f"{REMOTE_ROOT}/cube2_official_embeddings_v1/stat-freeze.json",
        "sha256": "53e5bd43e3054287ab9c6ad8a7ac79e954e2e028b984ea8cf0a8a54c1ec59360",
    }
    data["memory_estimate_bytes"] = {
        "steady_host_compact_arrays_breakdown": {
            "observations_FP32": 148148000,
            "actions_FP32": 20020000,
            "compact_terminals_FP32": 4004000,
            "compact_valids_FP32": 4004000,
            "valid_idxs_int64": 8008000,
            "GCDataset_terminal_and_initial_locs_int64": 35200,
        },
        "steady_host_compact_arrays_sum": 184683200,
        "visual_readonly_feature_payload_train_plus_val": 2254848000,
        "visual_steady_arrays_plus_features": 2439531200,
        "NPY_file_headers_total": 140800,
        "scope": "Arithmetic payload floor, not actual peakRSS/PSS or GPU/JIT reserve; loading transients/runtime extra",
    }
    cfg["input_contract"]["goal_dim"] = 518
    cfg["input_contract"]["goal_state"] = "objectXYZ6 in cube index order; no robotstate"
    cfg["input_contract"]["object_XYZ_indices"] = [[19, 20, 21], [28, 29, 30]]
    cfg["reward"]["object_count"] = 2
    cfg["GPU_execution"] = f"Root serial only after the fresh {GROUP} cache is accepted"
    cfg["output"] = f"{REMOTE_ROOT}/{CASE_NAME}/runs/{GROUP}"
    cfg["data_plan"] = f"{REMOTE_ROOT}/{CASE_NAME}/cache_v1/producer_plan.json"
    cfg["parent_frozen_config"] = {
        "path": f"{REMOTE_ROOT}/cube23_distance_gciql_v1/five_group_extension_v1/configs/G05_cube2_rgb.json",
        "sha256": G05_CONFIG_SHA,
        "original_not_modified": True,
    }
    cfg["group_id"] = GROUP
    cfg["priority_run_id"] = CASE_NAME
    cfg["memory_estimate_bytes"] = {
        "host_payload_floor": 2439531200,
        "feature_payload": 2254848000,
        "four_projected_input_arrays_at_B1024": 8617984,
        "not_actual_measured_peak": True,
    }
    cfg["encoder_binding"].update(
        asset_id=CUBE2_ASSET_ID,
        checkpoint=f"{CUBE2_ASSET}/weights/weights_epoch_10.pt",
        checkpoint_sha256=CUBE2_WEIGHT_SHA,
        adapter={
            "path": f"{REMOTE_ROOT}/{CASE_NAME}/cache_v1/encoder_adapter.py",
            "sha256": "PENDING_MATERIALIZED_SHA",
        },
    )
    return cfg


def native_shim(target: Path) -> str:
    text = f'''"""Frozen G05 registration used only by the Cube2 PLDM Native adapter."""
import hashlib
import json
from pathlib import Path

PATH = Path("{REMOTE_ROOT}/cube23_distance_gciql_v1/five_group_extension_v1/configs/G05_cube2_rgb.json")
raw = PATH.read_bytes()
assert hashlib.sha256(raw).hexdigest() == "{G05_CONFIG_SHA}"
GROUPS = {{"G05": json.loads(raw)}}
'''
    path = target / "functional_v1/cube2_successor_v1/visual_sr_cube2.py"
    write_exclusive(path, text.encode())
    return sha(text.encode())


def materialize(source_root: Path, target: Path) -> dict:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"target is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    records = materialize_cache(source_root, target)

    inputs = functional_inputs(source_root)
    runtime = functional_runtime(source_root)
    prepare = functional_prepare(source_root)
    shim_sha = native_shim(target)
    native = functional_native(source_root, shim_sha)
    audit = functional_audit(source_root)
    generated = {
        "functional_v1/pldm_inputs.py": inputs.encode(),
        "functional_v1/pldm_runtime.py": runtime.encode(),
        "functional_v1/prepare.py": prepare.encode(),
        "functional_v1/native_sr.py": native.encode(),
        "functional_v1/audit_native.py": audit.encode(),
    }
    for relative, raw in generated.items():
        write_exclusive(target / relative, raw)
    write_exclusive(target / "run_case.py", Path(__file__).with_name("run_case.py").read_bytes())

    cfg = config_template(source_root)
    cfg["encoder_binding"]["adapter"]["sha256"] = sha((target / "cache_v1/encoder_adapter.py").read_bytes())
    config_path = target / "functional_v1/configs/P2_cube2_pldm_reaudit_rgb.TEMPLATE.json"
    write_json_exclusive(config_path, cfg)

    source_freeze = json.loads(source(source_root, "functional_v1/SOURCE_FREEZE_v2.json"))
    functional_files = [
        "functional_v1/pldm_inputs.py",
        "functional_v1/pldm_runtime.py",
        "functional_v1/prepare.py",
        "functional_v1/native_sr.py",
        "functional_v1/audit_native.py",
        "functional_v1/cube2_successor_v1/visual_sr_cube2.py",
        "functional_v1/configs/P2_cube2_pldm_reaudit_rgb.TEMPLATE.json",
    ]
    shared_original_pins = {
        path: digest
        for path, digest in source_freeze["original_pins"].items()
        if "cube3_pldm_gciql_v1" not in path
    }
    for name in (
        "cache_v1/encoder_adapter.py",
        "cache_v1/reference/render_adapter.py",
        "cache_v1/producer_bridge.py",
        "cache_v1/audit_bridge.py",
        "cache_v1/prepare_plan.py",
        "cache_v1/source_overlay.py",
    ):
        shared_original_pins[f"{REMOTE_ROOT}/{CASE_NAME}/{name}"] = sha((target / name).read_bytes())
    shared_original_pins[f"{CUBE2_ASSET}/MANIFEST.json"] = CUBE2_MANIFEST_SHA
    shared_original_pins[f"{CUBE2_ASSET}/RUNTIME_ADDENDUM.json"] = CUBE2_ADDENDUM_SHA
    freeze = {
        "schema_version": 1,
        "actual_exit_code": 0,
        "case_name": CASE_NAME,
        "group_id": GROUP,
        "files": {
            str(Path(name).relative_to("functional_v1")): sha((target / name).read_bytes())
            for name in functional_files
        },
        "original_pins": shared_original_pins,
        "contracts": {
            "cube_count": 2,
            "state_dim": 37,
            "current_dim": 531,
            "goal_dim": 518,
            "latent_dim": 512,
            "max_episode_steps": 500,
            "training_updates": 1000000,
            "SR_episodes": 250,
        },
        "old_files_modified": False,
        "NN_or_training_started": False,
    }
    write_json_exclusive(target / "functional_v1/SOURCE_FREEZE_v1.json", freeze)

    outputs = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and not path.name.endswith(".pyc"):
            raw = path.read_bytes()
            outputs[str(path.relative_to(target))] = {"bytes": len(raw), "sha256": sha(raw)}
    result = {
        "schema_version": 1,
        "actual_exit_code": 0,
        "case_name": CASE_NAME,
        "group_id": GROUP,
        "source_root": str(source_root),
        "target_root": str(target),
        "source_hashes": SOURCE_HASHES,
        "records": records,
        "outputs": outputs,
        "NN_started": False,
        "real_cache_created": False,
        "training_started": False,
        "SR_started": False,
    }
    write_json_exclusive(target / "MATERIALIZATION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_root.resolve(), args.target_root.resolve())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
