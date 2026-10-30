"""SHA-pinned asset labels only; archive/transport verification bodies stay unchanged."""
import hashlib
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
PACK_SHA = 'bd4a240d9a39d0d7140f873de62ce3f760651ae2223598a68882f680f30e69d4'
COPY_SHA = 'bcb574502929e6c128806c7f714788d0cfe02f991592b9ae1f8024f6bec1e574'
ASSET = 'PCD_cache_closed_metadata'
ARCHIVE = 'Quad_PCD_cache_closed_metadata_v1.tar.gz'
PACK_EDITS = [
    ("dict(files=files, final_only=True, old_periodics_raw_WM_cache_excluded=True)",
     "dict(files=files, asset_kind='PCD_cache_closed_metadata', cache_payload_model_raw_excluded=True)", 1),
    ("f'{args.group}_final1M_closed.tar.gz'", "'Quad_PCD_cache_closed_metadata_v1.tar.gz'", 1),
    ('actual_exit_code=0, group_id=args.group', "actual_exit_code=0, asset_kind='PCD_cache_closed_metadata'", 1),
    ('only_final_policy_and_sidecar=True, no_periodic_WM_raw_or_cache_copy=True',
     'metadata_only=True, no_WM_model_raw_or_cache_payload_copy=True', 1),
]
COPY_EDITS = [
    ("parser.add_argument('--group', choices=('G11', 'G12'), required=True)",
     "parser.add_argument('--asset-kind', choices=('PCD_cache_closed_metadata',), default='PCD_cache_closed_metadata')", 1),
    ("remote = ROOT + f'/cube4_visual_gciql_v1/closed_training_backup_v2/{args.group}/completed_v1'",
     "remote = ROOT + '/cube4_visual_gciql_v1/closed_pcd_metadata_backup_v1/completed_v1'", 1),
    ("r['group_id']==group", "r['asset_kind']==asset_kind", 1),
    ('group={args.group!r}', 'asset_kind={args.asset_kind!r}', 1),
    ("f'{args.group}_final1M_closed.tar.gz'", "'Quad_PCD_cache_closed_metadata_v1.tar.gz'", 1),
    ('checkpoint_SHA_ledger.json', 'PCD_receipt_SHA_ledger.json', 1),
    ('group_id=args.group', 'asset_kind=args.asset_kind', 1),
    ('only_final_policy_and_sidecar=True', 'metadata_only=True', 1),
    ("prefix=f'quad_{args.group}_closed_'", "prefix=f'quad_{args.asset_kind}_closed_'", 1),
]


def load(kind):
    filename, expected, edits = (('closed_role.py', PACK_SHA, PACK_EDITS) if kind == 'pack'
                                 else ('copy_verified.py', COPY_SHA, COPY_EDITS))
    path = HERE.parent/'closed_backup_preparation_v2'/filename
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected
    original = raw.decode('utf-8'); effective = original
    for before, after, count in edits:
        assert effective.count(before) == count, (kind, before)
        effective = effective.replace(before, after)
    restored = effective
    for before, after, count in reversed(edits):
        assert restored.count(after) == count, (kind, after)
        restored = restored.replace(after, before)
    assert restored == original
    module = types.ModuleType('_PCD_metadata_'+kind)
    module.__file__ = str(path)
    exec(compile(effective, str(path), 'exec'), module.__dict__)
    module.parameter_bridge_proof = dict(source_sha256=expected,
        effective_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        exact_count_edits=len(edits), reverse_source_equals_original=True,
        only_asset_labels_namespace_receipt_semantics_changed=True)
    return module
