"""Reuse accepted archive and transport bodies with exact asset-label substitutions."""
import hashlib
from pathlib import Path
import types

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
Q=ROOT/'cube4_gciql_native_sr_v1'
OLD_PACK=ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/closed_sr_backup_v1/build_closed_v2.py'
PACK_SHA='0e32e8e9e743fa53af7bd6687cab88ff1353c85c322538588cf66796d7b91571'
OLD_COPY=ROOT/'cube4_visual_gciql_v1/closed_backup_preparation_v2/copy_verified.py'
COPY_SHA='bcb574502929e6c128806c7f714788d0cfe02f991592b9ae1f8024f6bec1e574'
ASSET='Quad_final1M_Native_SR_closed_evidence'
ARCHIVE='G11_G12_Quad_final1M_Native_SR_closed_v1.tar.gz'
PACK_EDITS=[("for mode in ('full', 'smoke')", "for mode in ('full', 'tiny')"),
 ('raw_formal_episodes=1000, smoke_episodes=4','raw_formal_episodes=500, smoke_episodes=2'),
 ("'G05_G06_G08_G09_M5_restored_closed_SR_v1.tar.gz'", "'G11_G12_Quad_final1M_Native_SR_closed_v1.tar.gz'"),
 ("save('receipt.json', dict(actual_exit_code=0, archive_path=", "save('receipt.json', dict(actual_exit_code=0, asset_kind='Quad_final1M_Native_SR_closed_evidence', archive_path=")]
COPY_EDITS=[("parser.add_argument('--group', choices=('G11', 'G12'), required=True)", "parser.add_argument('--asset-kind', choices=('Quad_final1M_Native_SR_closed_evidence',), default='Quad_final1M_Native_SR_closed_evidence')"),
 ("remote = ROOT + f'/cube4_visual_gciql_v1/closed_training_backup_v2/{args.group}/completed_v1'", "remote = ROOT + '/cube4_gciql_native_sr_v1/closed_SR_backup_v1/completed_v1'"),
 ("r['group_id']==group", "r['asset_kind']==asset_kind"),
 ('group={args.group!r}', 'asset_kind={args.asset_kind!r}'),
 ("f'{args.group}_final1M_closed.tar.gz'", "'G11_G12_Quad_final1M_Native_SR_closed_v1.tar.gz'"),
 ("'checkpoint_SHA_ledger.json', 'closure_readback.json'", "'closure_readback.json'"),
 ('group_id=args.group', 'asset_kind=args.asset_kind'),
 ('only_final_policy_and_sidecar=True', 'models_excluded=True, formal_episodes=500, diagnostic_episodes_excluded_from_SR=2'),
 ("prefix=f'quad_{args.group}_closed_'", "prefix=f'quad_{args.asset_kind}_closed_'"),
 ('no_NN_training_SR_or_cache_payload_copy=True', 'no_NN_training_or_SR_reexecution_or_cache_payload_copy=True')]


def load(kind,source=None):
    path=Path(source or (OLD_PACK if kind=='pack' else OLD_COPY));raw=path.read_bytes()
    expected,edits=(PACK_SHA,PACK_EDITS) if kind=='pack' else (COPY_SHA,COPY_EDITS)
    assert hashlib.sha256(raw).hexdigest()==expected
    original=raw.decode();effective=original
    for before,after in edits:
        assert effective.count(before)==1,(kind,before);effective=effective.replace(before,after)
    restored=effective
    for before,after in reversed(edits):
        assert restored.count(after)==1,(kind,after);restored=restored.replace(after,before)
    assert restored==original
    module=types.ModuleType('_accepted_Quad_SR_'+kind);module.__file__=str(path)
    exec(compile(effective,str(path),'exec'),module.__dict__)
    module.reuse_proof=dict(source_sha256=expected,effective_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        strict_asset_edits=len(edits),inverse_entire_source_exact=True,archive_hash_CRC_member_verify_body_unchanged=True)
    return module
