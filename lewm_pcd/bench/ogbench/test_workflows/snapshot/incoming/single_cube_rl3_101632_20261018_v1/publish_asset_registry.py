"""Publish immutable references to six Cube/LeWM assets after accepted import.

No feature decoding, raw scans, model construction, training or SR. Existing
Cube2/Cube3 assets and imported source snapshots are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

SOURCE_W='/root/autodl-tmp/workspace/users/lewm'
ARCHIVE_SHA='80d420173d2d594e1b3deaa2612d20f088ba0045ecfb4ac58f141f3e9ef3872a'
FORMAL='coordination/rl3_m0_m5_m6_visual_proprio_smoke_109_20261013_v1/formal_start_v1'
BASE='ogbench-official-single-play-20261006'
CACHE_BASE=BASE+'/official_lewm_gcbc_20261006/goal_xyz_mask_100k_b256_20261011_v1'
TERMINAL_BASE='coordination/cached_lewm_state_gciql_xyz_109_20261012_v1/terminal_v1/data_v2'
WM={
 (1,'rgb'):('ogbench-train-20261003/checkpoints/fixed5_mixed_pix_10ep_chunk5','88abffcda76ce55408639aa2c1ebeb2879f38d00dad6798cbb60bf30a03fe60d'),
 (1,'pcd'):('ogbench-train-20261003/checkpoints/fixed5_mixed_pcd_dexart_10ep','ab9f2e69d58eea4f4d8f469a7dd259c5944f45d0d54ec20061cf63c7a9630d03'),
 (2,'rgb'):('checkpoints/cube2/2d_epoch10','995a99f4904d87e71424b2de7910e7e0cf79556de8c2cebd08ac582d70880190'),
 (2,'pcd'):('checkpoints/cube2/3d_epoch10','f391f047aa73fa08b2b4534e6018fde6a219cc3ae6ba4464b907e2b9f9138a46'),
 (3,'rgb'):('checkpoints/2d_epoch10','4e16a6c5c58e8c44728d153e4ca7149d05676973379d99774ff7db2e08bbfd4e'),
 (3,'pcd'):('checkpoints/3d_epoch10','07bb8d2160b1a0fecda35bba3aede4790071540a6d7a9c34b92d89ae437a04b6')}
SINGLE_PINS={
 'rgb':('f42e4388750df90869e66a4f0a25c44f64b8074a8dc82ab0b2ec65e6deb84afe','c980135b0901843b71f995f0ea8a6ff8bb2453342b776ea771b9c40025d7abe3','7724d254892915e94fb483142e61850d024c2c63edbaf1eebf838269a1ee80a3'),
 'pcd':('7e05afd39bd163b3738776ac803d5caf812dfec8afe42a0ade10f73e13daf3aa','1b37c2700e0f5da387fd39ae5e939e3c9f6b3bfe16a80c4fdb8c097384fbcef6','41941bd7837e73aaa20c9d5f194a67a1aed86fc8b29b45a44f1e80753fe9b257')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def stat(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def read(path,expected=None):
    before=stat(path);content=Path(path).read_bytes()
    assert stat(path)==before
    digest=hashlib.sha256(content).hexdigest()
    assert expected is None or digest==expected,str(path)
    return json.loads(content),dict(path=str(path),sha256=digest,stat=before)


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def source_member(source):
    rel=PurePosixPath(source).relative_to(PurePosixPath(SOURCE_W))
    assert '..' not in rel.parts
    return str(PurePosixPath('source_layout')/rel)


class Imported:
    def __init__(self,root):
        self.root=root
        self.accepted,self.accepted_ref=read(root/'IMPORT_ACCEPTANCE.json')
        a=self.accepted
        assert a['actual_exit_code']==0 and a['archive_sha256']==ARCHIVE_SHA
        assert a['source_files_verified']==2439 and a['all_current_target_SHA_match_expected'] is True
        self.origin,self.origin_ref=read(root/'IMPORT_ORIGIN_MANIFEST.json')
        o=self.origin
        assert o['source_host']=='101632' and o['cube_count']==1 and o['data_id']=='cube-single-play-v0'
        assert o['source_root']==SOURCE_W and o['source_file_count']==2439
        assert o['all_current_source_SHA_match_original_acceptance'] is True
        self.sources={e['path']:e for e in o['files']}
        self.current={e['member']:e for e in a['files']}
        assert len(self.sources)==len(self.current)==2439

    def path(self,source,hash_content=False):
        entry=self.sources[source]
        member=source_member(source)
        assert entry['member']==member and member in self.current
        current=self.current[member]
        expected=entry['expected_sha256']
        assert entry['sha256']==current['sha256']==expected
        target=self.root/member
        assert target.resolve().is_relative_to(self.root.resolve()) and target.is_file()
        assert stat(target)==current['current_target_stat']
        assert current['bytes']==entry['bytes']==target.stat().st_size
        if hash_content:assert sha(target)==expected
        return target

    def relative(self,relative,hash_content=False):
        return self.path(SOURCE_W+'/'+relative,hash_content)


def world_model(cube,modality,root,imported):
    relative,expected=WM[(cube,modality)]
    directory=(imported.root/'source_layout'/relative) if cube==1 else root/relative
    weights=directory/'weights_epoch_10.pt'
    assert sha(weights)==expected
    configs={name:dict(path=str(directory/name),sha256=sha(directory/name),stat=stat(directory/name))
             for name in ('config.json','config.yaml')}
    if cube==1:
        for name in ('weights_epoch_10.pt','config.json','config.yaml'):
            assert imported.relative(relative+'/'+name).resolve()==(directory/name).resolve()
    return dict(epoch=10,checkpoint=str(weights),checkpoint_sha256=expected,
                checkpoint_stat=stat(weights),configs=configs)


def single_asset(modality,root,imported):
    relative=CACHE_BASE+f'/{modality}_latent_cache_20261012_v1/features_v1'
    directory=imported.root/'source_layout'/relative
    manifest_ref=imported.relative(relative+'/cache_manifest.json',True)
    seal_ref=imported.relative(relative+'/cache_seal.json',True)
    term_relative=TERMINAL_BASE+'/'+modality+'/terminal_manifest.json'
    terminal_ref=imported.relative(term_relative,True)
    manifest,mm=read(manifest_ref,SINGLE_PINS[modality][0])
    seal,sm=read(seal_ref,SINGLE_PINS[modality][1])
    terminal,tm=read(terminal_ref,SINGLE_PINS[modality][2])
    expected=WM[(1,modality)][1]
    assert manifest['source_checkpoint_sha256']==terminal['source_checkpoint_sha256']==expected
    assert manifest['exit_code']==seal['exit_code']==terminal['exit_code']==0
    assert seal['manifest_sha256']==SINGLE_PINS[modality][0]
    assert terminal['cache_manifest_sha256']==SINGLE_PINS[modality][0]
    assert terminal['cache_seal_sha256']==SINGLE_PINS[modality][1]
    wm=world_model(1,modality,root,imported)
    assert wm['configs']['config.yaml']['sha256']==manifest['source_config_sha256']
    for split,count in (('train',1000),('val',100)):
        entries=manifest['splits'][split]
        assert [e['episode'] for e in entries]==list(range(count))
        for entry in entries:
            assert entry['shape']==[1000,192] and entry['dtype']=='float32'
            assert imported.sources[entry['path']]['expected_sha256']==entry['sha256']
            imported.path(entry['path'])
        terminal_file=terminal['splits'][split]
        assert terminal_file['shape']==[count,192] and terminal_file['dtype']=='float32'
        assert imported.sources[terminal_file['path']]['expected_sha256']==terminal_file['sha256']
        imported.path(terminal_file['path'])
    return dict(cube_count=1,data_id='cube-single-play-v0',modality=modality,
        cache_directory=str(directory),layout='main1000_npy_plus_separate_true_terminal_npz',
        latent_dim=192,dtype='float32',split_episodes=dict(train=1000,val=100),
        canonical_raw_rows_per_episode=1001,canonical_total_rows=1101100,
        raw_id_mapping=dict(main='episode*1001+step, step0..999',terminal='episode*1001+1000'),
        manifest=mm,seal=sm,terminal_manifest=tm,world_model=wm,
        preprocess_source_sha256=manifest['preprocess_source_sha256'],
        producer_source_sha256=manifest['producer_source_sha256'],
        old_manifest_paths_and_inode_are_provenance_only=True,
        current_identity='IMPORT_ACCEPTANCE full member SHA plus fresh target stat; original manifest bytes retained',
        new_vector_SHA_scan=False,full_finite_scan=False)


def existing_asset(cube,modality,root,imported):
    mode={'rgb':'2d','pcd':'3d'}[modality]
    directory=root/(f'cube2_official_embeddings_v1/{mode}/cache' if cube==2 else f'official_state_{mode}_cache_v1/cache')
    count=1000 if cube==2 else 3000
    ready,rr=read(directory/'READY.json')
    assert ready['ready_for_downstream'] is True
    if cube==2:
        audit_path,audit_sha=ready['audit_report'],ready['audit_sha256']
    elif modality=='rgb':
        audit_path,audit_sha=ready['audit_report_path'],ready['audit_report_sha256']
    else:
        audit_path,audit_sha=ready['feature_report_path'],ready['feature_report_sha256']
    audit,ar=read(audit_path,audit_sha)
    assert audit['actual_exit_code']==0 and audit['episodes_verified']==count+count//10
    assert audit['raw_rows_verified']==(count+count//10)*1001
    manifest,mm=read(directory/'cache_manifest.json',ready['manifest_sha256'])
    seal,sm=read(directory/'cache_seal.json')
    assert seal['actual_exit_code']==manifest['actual_exit_code']==0
    assert seal['manifest_sha256']==mm['sha256']
    expected=WM[(cube,modality)][1]
    assert manifest['checkpoint_sha256']==expected
    wm=world_model(cube,modality,root,imported)
    for config in ('config.json','config.yaml'):
        relative=WM[(cube,modality)][0]+'/'+config
        assert wm['configs'][config]['sha256']==manifest['protocol']['source_pins'][relative]
    tracked=hashlib.sha256()
    for split,episodes in (('train',count),('val',count//10)):
        rows=manifest['splits'][split]
        assert [entry['episode'] for entry in rows]==list(range(episodes))
        for entry in rows:
            receipt,ref=read(entry['receipt_path'],entry['receipt_sha256'])
            latent=receipt['latent']
            assert latent['shape']==[1001,192] and latent['dtype']=='float32'
            value=Path(latent['path']);assert value.resolve().is_relative_to(directory.resolve())
            current=stat(value);assert current['bytes']==latent['bytes']
            tracked.update(json.dumps(dict(path=str(value),stat=current),sort_keys=True).encode())
    exit_path=ready.get('outer_exit_file_required') or ready.get('final_wrapper_exit_required')
    if exit_path:assert Path(exit_path).read_text().strip()=='0'
    return dict(cube_count=cube,data_id=f"cube-{'double' if cube==2 else 'triple'}-play-v0",modality=modality,
        cache_directory=str(directory),layout='raw1001_rows_per_episode_single_npy',latent_dim=192,dtype='float32',
        split_episodes=dict(train=count,val=count//10),canonical_raw_rows_per_episode=1001,
        canonical_total_rows=(count+count//10)*1001,raw_id_mapping='episode*1001+step, step0..1000 including true terminal',
        manifest=mm,seal=sm,ready=rr,accepted_audit=ar,world_model=wm,
        preprocess_source_sha256=manifest['protocol']['source_pins']['source/lewm_pcd/utils.py'],
        current_metadata_stat_fingerprint=tracked.hexdigest(),new_vector_SHA_scan=False,
        current_check_scope='small manifest/receipt SHA and feature size/stat only; prior complete audits preserved',
        original_assets_modified=False)


def gciql_refs(root,imported):
    relative=FORMAL
    original=imported.root/'source_layout'/relative
    for directory in ('core_v1','configs','native1m_v1'):
        assert (original/directory).is_dir(),directory
    roles={}
    for role,worker,current in (('M0',2,220),('M5',4,211),('M6',4,211)):
        runrel=BASE+f'/rl3_{role.lower()}_gciql_seed0_b1024_1m_w{worker}_20261013_v1'
        run=imported.root/'source_layout'/runrel
        resolved_path=imported.relative(runrel+'/resolved.json',True)
        resolved,resref=read(resolved_path)
        config=resolved['binding']['config']
        assert config['total_updates']==1000000 and config['workers']==worker
        assert resolved['binding']['agent_config']['alpha']==1.0
        weights=run/'checkpoints/update_001000000/params_1000000.pkl'
        source=SOURCE_W+'/'+runrel+'/checkpoints/update_001000000/params_1000000.pkl'
        imported.path(source,True)
        native_rel=BASE+f'/rl3_{role.lower()}_gciql_1m_b1024_native_20261013_v1'
        native=imported.root/'source_layout'/native_rel
        assert native.is_dir()
        roles[role]=dict(cube_count=1,training_run=str(run),Native_SR=str(native),resolved=resref,
            checkpoint=str(weights),checkpoint_sha256=sha(weights),current_dim=current,goal_dim=195,
            old_source_config_snapshots_unmodified=True)
    return dict(cube_count=1,algorithm='GCIQL',route='RL3 visualproprio reference, not new GCIQL algorithmv2',
        formal_root=str(original),roles=roles,training_reward='custom current object distance4cm, not official source-index equality',
        input_contract='M5/M6 cached192+proprio19=211; goalcached192+XYZ3=195; M0zero192+fullstate28=220',
        runnable_on_target=False,reason='Historical source absolute paths/runtime bindings retained; path map is provenance, no launch adapter prepared',
        training_SR_or_nn_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--import-root',type=Path,required=True)
    parser.add_argument('--registry-out',type=Path,required=True)
    args=parser.parse_args()
    root,import_root=args.root.resolve(),args.import_root.resolve()
    registry=args.registry_out.resolve()
    assert import_root==root/'assets/imports/single_cube_rl3_101632_20261018_v1'
    assert registry==root/'assets/registry/single_cube_rl3_import_20261018_v1.json'
    imported=Imported(import_root)
    entries=[single_asset(modality,root,imported) if cube==1 else existing_asset(cube,modality,root,imported)
             for cube in (1,2,3) for modality in ('rgb','pcd')]
    downstream=gciql_refs(root,imported)
    group=root/'assets/downstream_gciql/cube1_rl3_reference_v2'
    aliases=[root/f"assets/embeddings/cube{e['cube_count']}/{e['modality']}" for e in entries]
    assert not registry.exists() and not group.exists() and all(not p.exists() for p in aliases), 'No overwrite'
    path_map=dict(source_root=SOURCE_W,target_source_layout=str(import_root/'source_layout'),
        original_snapshots_modified=False,launch_ready=False,
        copied_source_paths={p:str(import_root/source_member(p)) for p in imported.sources})
    group.mkdir(parents=True)
    write_new(group/'PATH_MAP.json',path_map)
    write_new(group/'REFERENCE_GROUP.json',downstream)
    for name,source in [('core',Path(downstream['formal_root'])/'core_v1'),
                        ('configs',Path(downstream['formal_root'])/'configs'),
                        ('native_reference',Path(downstream['formal_root'])/'native1m_v1')]:
        (group/name).symlink_to(source,target_is_directory=True)
    for alias,entry in zip(aliases,entries):
        alias.mkdir(parents=True)
        write_new(alias/'ASSET_REF.json',dict(entry,reference_only=True))
        (alias/'cache').symlink_to(entry['cache_directory'],target_is_directory=True)
        (alias/'world_model').symlink_to(Path(entry['world_model']['checkpoint']).parent,target_is_directory=True)
    write_new(registry,dict(schema_version=1,actual_exit_code=0,completed=time.time(),
        source_import=imported.origin_ref,import_acceptance=imported.accepted_ref,
        embeddings=entries,downstream_group=str(group),downstream=downstream,
        asset_identity_fields=['cube_count','data_id','modality','world_model.checkpoint_sha256','manifest.sha256','raw_id_mapping','layout'],
        source_manifest_bytes_preserved=True,old_source_inodes_not_used_as_target_identity=True,
        current_import_stat_rebased=True,original_Cube2_Cube3_modified=False,
        vectors_decoded_or_reencoded=False,NN_training_or_SR_executed=False,
        runnable_on_target=False))
    print(json.dumps(dict(actual_exit_code=0,registry=str(registry),registry_sha256=sha(registry),
                         embedding_entries=len(entries),downstream_group=str(group))))


if __name__=='__main__':
    main()
