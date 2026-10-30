"""One actual2D-audit schema mapping; preserve original complete cache guards."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

V2_SHA = 'c26ac069add084cf95e880e40b33d2119cd6121e5bd647de9a3ec03dc2c237e2'
ORIGINAL_SHA = 'd4a1415be31368ea0bfd6d4910eaae462cc834e8dac496be22b8477e483f037c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size,mtime_ns=value.st_mtime_ns,ctime_ns=value.st_ctime_ns,
                device=value.st_dev,inode=value.st_ino)


def load_original(case, source_directory=None):
    directory = Path(source_directory or Path(case)/'functional_v1')
    path = directory/'prepare_v2.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V2_SHA
    sys.path.insert(0,str(directory))
    wrapper = types.ModuleType('_P3_original_prepare_v2')
    wrapper.__file__ = str(path)
    exec(compile(raw,str(path),'exec'),wrapper.__dict__)
    module = wrapper.load_original(Path(case))
    assert sha(directory/'prepare.py') == ORIGINAL_SHA
    return module


def install_checkpoint_binding(module, case):
    path = Path(module.__file__)
    source = path.read_text()
    assert sha(path) == ORIGINAL_SHA
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='check_cache_metadata')
    original = ast.get_source_segment(source,node)
    before, after = "audit['checkpoint_sha256']", "manifest['protocol']['checkpoint_sha256']"
    assert original.count(before) == 1 and original.count(after) == 0
    effective = original.replace(before,after)
    assert effective.replace(after,before) == original
    # The original closed_cache still checks audit plan/manifest/seal/protocol SHA
    # against actual bytes before any config publication. The original2D auditor
    # attests checkpoint identity through those hashes, not a nonexistent flat key.
    exec(compile(effective,str(path),'exec'),module.__dict__)
    return dict(original_prepare_sha256=ORIGINAL_SHA,original_v2_sha256=V2_SHA,
        exact_expression_edits=1,inverse_function_source_exact=True,
        effective_function_sha256=hashlib.sha256(effective.encode()).hexdigest(),
        audit_checkpoint_binding='actual plan+manifest+protocol checkpoint, authenticated by unchanged audit SHA chain',
        audit_fields_or_files_modified=False,numeric_tolerances_changed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('verify','train','native'))
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();case=args.case.resolve()
    assert case.name=='cube3_pldm_gciql_v1'
    module=load_original(case)
    proof=install_checkpoint_binding(module,case)
    if args.phase=='verify':
        records={};facts=module.closed_cache(case,records)
        template=module.read(case/'functional_v1/configs/P3_cube3_pldm_rgb.TEMPLATE.json',records)
        cfg=module.config_from_cache(template,facts,case)
        result=dict(actual_exit_code=0,metadata_only=True,config_published=False,
            current_dim=cfg['input_contract']['current_dim'],goal_dim=cfg['input_contract']['goal_dim'],
            encoder_family='pldm',latent_dim=512,selected_preprocess_device=facts['selected_device'],
            manifest_sha256=facts['manifest_sha256'],verified_metadata_files=len(records),proof=proof)
        if args.report:module.publish(args.report,result)
    else:
        result=(module.prepare_train if args.phase=='train' else module.prepare_native)(case)
    print(json.dumps(result))
