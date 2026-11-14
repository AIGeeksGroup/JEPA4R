# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/foundation_assets.py (verbatim except: imports / paths adapted
# to the package; lines wrapped at 120 columns)
"""Plan/prepare released encoder bytes and fresh bindings; never load a model."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile

HERE=Path(__file__).resolve().parent
LEWM=HERE.parents[2]   # this repository
VJEPA2=Path(os.environ.get('JEPA4R_VJEPA2') or LEWM/'third_party/vjepa2')   # V-JEPA 2 204698b checkout


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return digest.hexdigest()


def asset(model):
    registry=json.loads((LEWM/'configs/method/cube/released_models.json').read_text(encoding='utf8'))['models']
    if model not in ('visreg','vjepa'):
        raise ValueError('Released model must be visreg or vjepa')
    return copy.deepcopy(registry[model])


def _save(path,value,*,exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf8') as stream:
            json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
    else:
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf8')
        temporary.replace(path)


def build_plan(model,cube_count,mode,assets_root,output_root,weights=None):
    if type(cube_count) is not int or cube_count not in (1,2,3,4) or mode not in ('smoke','formal'):
        raise ValueError('Explicit Cube1..4 and smoke/formal mode required')
    entry=asset(model)
    assets_root,output_root=Path(assets_root).resolve(),Path(output_root).resolve()
    resolved=Path(weights).resolve() if weights is not None else assets_root/model/entry['filename']
    if resolved.is_relative_to(output_root):
        raise ValueError('Receipt output must be separate from weight storage')
    return dict(schema='cube_released_asset_plan_v1',model=model,cube_count=cube_count,mode=mode,
                asset=entry,assets_root=str(assets_root),output_root=str(output_root),
                explicit_weights=str(Path(weights).resolve()) if weights is not None else None,
                weights=str(resolved),execute=False,numeric_acceptance=False,model_loaded=False)


def download(plan):
    """Explicit owned-partial resume; complete existing files are never replaced."""
    target=Path(plan['weights']);entry=plan['asset']
    if plan['explicit_weights'] is not None:
        raise ValueError('Automatic download may only write the dedicated assets-root target')
    if target.exists():
        raise FileExistsError('Preserve completed weights: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_name(target.name+'.part')
    owner=target.with_name(target.name+'.request.json')
    status=target.with_name(target.name+'.download.json')
    request=dict(model=plan['model'],url=entry['url'],bytes=entry['bytes'],sha256=entry['sha256'],target=str(target))
    if owner.exists():
        if json.loads(owner.read_text())!=request:
            raise ValueError('Existing partial belongs to another release')
    else:
        if part.exists():
            raise FileExistsError('Unowned partial must be inspected, not overwritten')
        _save(owner,request,exclusive=True)
    offset=part.stat().st_size if part.exists() else 0
    if offset>entry['bytes']:
        raise ValueError('Owned partial exceeds released byte count')
    state=dict(started_at=time.time(),actual_exit_code=None,completed_bytes=offset,request=request)
    _save(status,state)
    try:
        if offset<entry['bytes']:
            headers={'User-Agent':'JEPA4R-released-assets/1'}
            if offset:headers['Range']='bytes='+str(offset)+'-'
            with urllib.request.urlopen(urllib.request.Request(entry['url'],headers=headers),timeout=60) as response:
                code=getattr(response,'status',200)
                if offset and (
                    code != 206
                    or not response.headers.get('Content-Range', '').startswith('bytes ' + str(offset) + '-')
                ):
                    raise ValueError('Server did not honor owned partial Range; preserve partial')
                with part.open('ab' if offset else 'xb') as stream:
                    for block in iter(lambda:response.read(8<<20),b''):
                        stream.write(block);offset+=len(block)
                        if offset>entry['bytes']:
                            raise ValueError('Download exceeds exact release size')
                        state.update(completed_bytes=offset,updated_at=time.time());_save(status,state)
                    stream.flush();os.fsync(stream.fileno())
        if offset!=entry['bytes'] or sha(part)!=entry['sha256']:
            raise ValueError('Downloaded release bytes/SHA mismatch; partial retained')
        if target.exists():raise FileExistsError(target)
        part.rename(target)
        state.update(actual_exit_code=0,completed_bytes=offset,completed_at=time.time())
        _save(status,state)
    except BaseException as error:
        state.update(actual_exit_code=1,error=repr(error),updated_at=time.time());_save(status,state)
        raise
    return target


def _source_binding(entry):
    files={str(path):sha(path)
           for path in (Path(__file__),HERE/'released_encoders.py',LEWM/'configs/method/cube/released_models.json')}
    if 'source_files_sha256' in entry:
        root=VJEPA2
        for relative,digest in entry['source_files_sha256'].items():
            path=root/relative
            if not path.is_file() or sha(path)!=digest:
                raise ValueError('Released official model source missing/changed: '+str(path))
            files[str(path)]=digest
    return files


def prepare(plan,download=False):
    expected = build_plan(
        plan['model'],
        plan['cube_count'],
        plan['mode'],
        plan['assets_root'],
        plan['output_root'],
        plan['explicit_weights'],
    )
    if plan!=expected:
        raise ValueError('Released asset plan differs from the selected immutable release')
    output=Path(plan['output_root'])
    if output.exists():raise FileExistsError(output)
    path=Path(plan['weights']);entry=plan['asset']
    if not path.is_file():
        if not download:
            raise FileNotFoundError('Released weights missing; provide --weights or explicitly --download: '+str(path))
        path=globals()['download'](plan)
    before=path.stat()
    if before.st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
        raise ValueError('Released checkpoint bytes/SHA/hash differs: '+str(path))
    if not zipfile.is_zipfile(path):raise ValueError('Released checkpoint is not a complete Torch ZIP archive')
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('Released checkpoint archive CRC failed')
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('Released checkpoint changed during byte validation')
    sources=_source_binding(entry)
    spec={key:copy.deepcopy(entry[key]) for key in ('family','native_size','feature','inference_precision')}
    spec.update(checkpoint=str(path),checkpoint_sha256=entry['sha256'],output_dim=entry['latent_dim'],
                diagnostic_only=False,official_pretrained=True)
    for key in (
        'source_commit',
        'source_files_sha256',
        'state_dict_key',
        'encoder_module',
        'encoder_constructor',
        'constructor_kwargs',
        'timm_version',
    ):
        if key in entry:spec[key]=copy.deepcopy(entry[key])
    if 'source_commit' in entry:spec['source_root']=str(VJEPA2)
    output.mkdir(parents=True,exist_ok=False)
    spec_path=output/'encoder_spec.json';_save(spec_path,spec,exclusive=True)
    adapter=HERE/'released_encoders.py'
    binding = dict(
        family=plan['model'],
        latent_dim=entry['latent_dim'],
        checkpoint=str(path),
        checkpoint_sha256=entry['sha256'],
        adapter=dict(path=str(adapter), sha256=sha(adapter)),
        spec=dict(path=str(spec_path), sha256=sha(spec_path)),
        preprocessing=dict(
            source_path=str(adapter), source_sha256=sha(adapter), device='cuda', input_dtype='uint8', layout='HWC224'
        ),
    )
    receipt = dict(
        schema='cube_workflow_encoder_v1',
        model=plan['model'],
        cube_count=plan['cube_count'],
        mode=plan['mode'],
        actual_exit_code=0,
        checkpoint=dict(path=str(path), sha256=entry['sha256']),
        checkpoint_bytes=entry['bytes'],
        released_weights_verified=True,
        strict_reload_verified=False,
        model_loaded=False,
        numeric_acceptance=False,
        verification_scope='exact released checkpoint size/SHA256/archive CRC and official source files; no model '
                           'import',
        published_official_sha256=entry.get('published_official_sha256', False),
        source_url=entry['url'],
        source_binding=sources,
        encoder_binding=binding,
        created_at=time.time(),
    )
    receipt_path=output/'ENCODER.json';_save(receipt_path,receipt,exclusive=True)
    _save(output/'actual_exit_code.json',dict(actual_exit_code=0),exclusive=True)
    return dict(receipt_path=str(receipt_path),encoder_binding=binding,model_loaded=False,numeric_acceptance=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('visreg','vjepa'),required=True)
    parser.add_argument('--cube-count',type=int,choices=(1,2,3,4),required=True)
    parser.add_argument('--mode',choices=('smoke','formal'),required=True)
    parser.add_argument('--assets-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(argv)
    plan=build_plan(args.model,args.cube_count,args.mode,args.assets_root,args.output,args.weights)
    print(json.dumps(plan,indent=2))
    if args.execute:print(json.dumps(prepare(plan,download=args.download),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
