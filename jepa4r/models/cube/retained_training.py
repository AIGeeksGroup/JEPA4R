# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/retained_training.py (verbatim except: imports / paths of the
# package layout, lines wrapped at 120 columns, no lint exemptions, unused imports removed, registration imports as
# importlib.import_module)
"""Explicit content-preserving relocation of the retained three-shard Cube1 input."""
import importlib
import argparse,copy,hashlib,json
from pathlib import Path

MISSING_METADATA={'LICENSE':'ff0b00150aa5d2ad4e4658a5f7b90d9316f9baa2199429255e61e3a47ee54252',
 'pyproject.toml':'e69329cff17f0090e411f79f45053f2289c9d01c9d6efc0d120525cbee0d9f20'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(8<<20),b''):h.update(data)
    return h.hexdigest()
def pin(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def pinned(record):
    path=Path(record['path']).resolve()
    if sha(path)!=record['sha256']:raise ValueError('retained file checksum differs: '+str(path))
    return path
def write(path,value):
    with Path(path).open('x',encoding='utf8')as f:json.dump(value,f,indent=2,allow_nan=False)

def content_aliases(original,available):
    by_sha={}
    for path in available:
        path=Path(path).resolve();by_sha.setdefault(sha(path),path)
    mapping={};missing=[]
    for name,digest in original.items():
        if digest in by_sha:mapping[name]=dict(path=str(by_sha[digest]),sha256=digest)
        elif any(name.endswith('/lewm_pcd/third_party/stable-worldmodel/'+base)and digest==expected
                 for base,expected in MISSING_METADATA.items()):
            missing.append(dict(original_path=name,sha256=digest,scope='unavailable original non-executable metadata'))
        else:raise ValueError('unmapped original producer file: '+name)
    return mapping,missing

def original_contract(old):
    expected=dict(schema='cube_fixed_collection_v1',cube_count=1,mode='smoke',raw_horizon=200,
                  collection_seed=0,train_episodes=2,validation_episodes=1,actual_exit_code=0,
                  protocol='paper_cube_fixed_mix_v1')
    if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('explicit retained Cube1 2/1 input required')
    rows=old.get('records',[])
    if len(rows) != 3 or [(r['part']['index'], r['part']['split'], r['part']['episodes']) for r in rows] != [
        (0, 'train', 1),
        (1, 'train', 1),
        (2, 'validation', 1),
    ]:
        raise ValueError('original retained shard order/split differs')
    seeds=[seed for r in rows for seed in r['qc']['reset_seeds']]
    initials=[value for r in rows for value in r['qc']['initial_state_sha256']]
    if len(set(seeds))!=3 or len(set(initials))!=3 or any(r['qc'].get('actual_exit_code')!=0 for r in rows):
        raise ValueError('original retained QC/seed/initial-state identity differs')

def validate_manifest(value,manifest_path):
    desc=read(pinned(value['retained_import']))
    if desc.get('schema')!='cube_retained_training_import_v1' or desc.get('scope')!='retained_bytes_no_new_collection':
        raise ValueError('retained training descriptor scope differs')
    old=read(pinned(desc['original_receipt']));original_contract(old)
    if desc['original_producer_binding']!=old['producer_binding']:
        raise ValueError('original producer identity was changed during import')
    mapping, missing = content_aliases(
        old['producer_binding']['files'], [pinned(row) for row in desc['producer_aliases'].values()]
    )
    if mapping!=desc['producer_aliases'] or missing!=desc['unavailable_non_executable_metadata']:
        raise ValueError('original producer aliases/metadata availability differs')
    if value['producer_binding']!=desc['resolved_producer_binding']:
        raise ValueError('training manifest identifies another retained producer')
    rows=desc['raw_shards']
    if [(r['index'],r['modality'])for r in rows]!=[(i,m)for i in range(3)for m in ('rgb','pcd')]:
        raise ValueError('retained raw shard order/modality differs')
    for row in rows:
        actual=pinned(row['actual'])
        if row['original_path']!=old['records'][row['index']]['part'][row['modality']]:
            raise ValueError('retained raw split/part identity differs')
        if old['data_files'][row['original_path']]['sha256']!=row['actual']['sha256']:
            raise ValueError('retained raw differs from its original receipt')
        recorded=value['source_files'].get(str(actual),{})
        if recorded.get('sha256')!=row['actual']['sha256']:
            raise ValueError('training manifest raw dependency identity differs')
    if value['raw']!=desc['derived_raw'] or value['collection_protocol']!=old['collection_protocol']:
        raise ValueError('retained raw paths or collection protocol changed')
    derived=read(value['collection_manifest'])
    if (derived['retained_original_receipt']!=desc['original_receipt']or
            derived['original_producer_binding']!=old['producer_binding']or
            derived['producer_binding']!=desc['resolved_producer_binding']or derived['records']!=old['records']):
        raise ValueError('derived collection view lost the original identity')
    _validate_vds_order(value,desc)
    return desc

def _validate_vds_order(value,desc):
    """Inspect every VDS mapping, including exactly which terminal rows it drops."""
    import h5py
    for split,indices in [('train',(0,1)),('validation',(2,))]:
        for modality in ('rgb','pcd'):
            key=split+'_'+modality
            sources=[r['actual']['path']for r in desc['raw_shards']if r['index']in indices and r['modality']==modality]
            raw=Path(value['raw'][key]).resolve()
            with h5py.File(raw,'r')as file:
                columns = []
                file.visititems(
                    lambda name, item: (
                        columns.append(name)
                        if isinstance(item, h5py.Dataset) and name not in ('ep_len', 'ep_offset')
                        else None
                    )
                )
                for name in columns:
                    maps=file[name].virtual_sources()
                    if len(maps)!=len(sources):raise ValueError('retained merged VDS source count differs')
                    for index,(mapping,source)in enumerate(zip(maps,sources)):
                        if (Path(mapping.file_name).resolve()!=Path(source).resolve()or mapping.dset_name!=name or
                            mapping.vspace.get_select_bounds()[0][0]!=201*index or
                            mapping.vspace.get_select_bounds()[1][0]!=201*index+200):
                            raise ValueError('retained merged VDS row/split order differs')
            with h5py.File(value['views'][key]['path'],'r')as file:
                for name in value['views'][key]['keys']:
                    maps=file[name].virtual_sources()
                    if len(maps)!=len(indices):raise ValueError('retained training VDS episode count differs')
                    for index,mapping in enumerate(maps):
                        if (Path(mapping.file_name).resolve()!=raw or mapping.dset_name!=name or
                            mapping.vspace.get_select_bounds()[0][0]!=200*index or
                            mapping.vspace.get_select_bounds()[1][0]!=200*index+199 or
                            mapping.src_space.get_select_bounds()[0][0]!=201*index or
                            mapping.src_space.get_select_bounds()[1][0]!=201*index+199):
                            raise ValueError('retained training VDS terminal-row mapping differs')

def create_binding(
    collection_root,
    baseline_root,
    output,
    *,
    expected_receipt_sha256,
    expected_baseline_manifest_sha256,
    metadata_files=(),
):
    importlib.import_module('hdf5plugin')  # Original HDF compression, no data rewrite.
    from jepa4r.data.cube import fresh_data
    from jepa4r.data.cube.collection import merge_raw
    collection,baseline,out=map(lambda p:Path(p).resolve(),(collection_root,baseline_root,output))
    receipt=collection/'COLLECTION.json';source=baseline/'SOURCE_MANIFEST.json'
    if sha(receipt)!=expected_receipt_sha256 or sha(source)!=expected_baseline_manifest_sha256:
        raise ValueError('trusted retained receipt/baseline manifest pin differs')
    old=read(receipt);original_contract(old)
    available=[]
    for name,h in read(source)['files'].items():
        path=(baseline/name).resolve()
        if not path.is_relative_to(baseline)or sha(path)!=h:raise ValueError('frozen baseline source changed')
        available.append(path)
    available += [p for p in collection.rglob('*')if p.is_file()]+list(metadata_files)
    mapping,missing=content_aliases(old['producer_binding']['files'],available)
    raw=[]
    for index in range(3):
        for suffix in ('pix','pcd'):
            relative=f'shards/part_{index:03d}/datasets/pair_{suffix}.h5'
            matches=[name for name in old['data_files']if name.endswith('/'+relative)]
            if len(matches)!=1:raise ValueError('original raw identity is ambiguous')
            bound=pin(collection/relative)
            if bound['sha256']!=old['data_files'][matches[0]]['sha256']:raise ValueError('original raw bytes differ')
            raw.append(dict(index=index,modality='rgb'if suffix=='pix'else'pcd',original_path=matches[0],actual=bound))
    out.mkdir(parents=True,exist_ok=False);(out/'raw_views').mkdir()
    paths={}
    for split,indices in [('train',(0,1)),('validation',(2,))]:
        for modality in ('rgb','pcd'):
            inputs=[row['actual']['path']for row in raw if row['index']in indices and row['modality']==modality]
            target=out/'raw_views'/f'{split}_{modality}.h5'
            merge_raw(inputs,target);paths[split+'_'+modality]=str(target)
    producer=copy.deepcopy(old['producer_binding'])
    producer.update(files={row['path']:row['sha256']for row in mapping.values()},
                    scope='same-byte aliases of original retained producer; not new collection')
    desc=dict(schema='cube_retained_training_import_v1',scope='retained_bytes_no_new_collection',
        original_receipt=pin(receipt),baseline_manifest=pin(source),original_producer_binding=old['producer_binding'],
        producer_aliases=mapping,unavailable_non_executable_metadata=missing,resolved_producer_binding=producer,
        raw_shards=raw,derived_raw=paths,importer=pin(__file__),original_data_modified=False)
    descriptor=out/'TRAINING_INPUT_IMPORT.json';write(descriptor,desc)
    derived=copy.deepcopy(old)
    derived.update(paths=paths,producer_binding=producer,original_producer_binding=old['producer_binding'],
        retained_original_receipt=pin(receipt),provenance_scope='retained bytes; new VDS descriptors only',
        data_files={row['actual']['path']:dict(sha256=row['actual']['sha256'])for row in raw})
    derived_path=out/'RETAINED_COLLECTION_VIEW.json';write(derived_path,derived)
    manifest=fresh_data.create_binding(derived_path,out/'training_views')
    value=read(manifest);value['retained_import']=pin(descriptor)
    fresh_data.save(manifest,value)  # This operation's newly created, unpublished descriptor only.
    fresh_data.load_binding(manifest)
    write(out/'IMPORT_RESULT.json',dict(actual_exit_code=0,actual_collection_episodes=0,actual_optimizer_updates=0,
        original_receipt=pin(receipt),training_manifest=pin(manifest),original_raw_files=len(raw),
        mapped_producer_files=len(mapping),unavailable_non_executable_metadata=missing,scope=desc['scope']))
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-receipt-sha256',required=True)
    p.add_argument('--expected-baseline-manifest-sha256', required=True)
    p.add_argument('--metadata-file', type=Path, action='append', default=[])
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    print(json.dumps(dict(scope='retained_bytes_no_new_collection',execute=args.execute,raw_shards=3,model_updates=0)))
    if args.execute:
        create_binding(
            args.collection_root,
            args.baseline_root,
            args.output,
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_baseline_manifest_sha256=args.expected_baseline_manifest_sha256,
            metadata_files=args.metadata_file,
        )
    return 0
if __name__=='__main__':raise SystemExit(main())
