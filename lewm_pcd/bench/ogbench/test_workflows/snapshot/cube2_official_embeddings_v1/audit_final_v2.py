"""CPU-only path-binding repair over the immutable Double feature auditor."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CASE=Path(__file__).resolve().parent
SHADOW=CASE/'workspace'
AUDITOR=SHADOW/'official_state_3d_cache_v1/audit_cache.py'
AUDITOR_SHA='c230f64373fb6f90f47d73b366688dafcf296fbc9361953070d92c905f8a554a'
PLAN_SHA='4010289ec8a4d8996cb656a32a4c0b434880c9ab59b6a12840b816b3641bed67'
CONFIG_REL='checkpoints/cube2/3d_epoch10/config.json'
BEFORE='    assert config_path.relative_to(workspace).as_posix() in pins'
AFTER="""    config_rel = 'checkpoints/cube2/3d_epoch10/config.json'
    config_matches = [relative for relative in pins if relative == config_rel
                      and (workspace / relative).resolve() == config_path]
    assert config_matches == [config_rel], 'Exact Double config pin binding'
    assert sha(config_path) == pins[config_rel], 'Resolved config SHA differs from original pin'"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch(source):
    assert source.count(BEFORE)==1
    effective=source.replace(BEFORE,AFTER)
    assert effective.count(AFTER)==1 and effective.replace(AFTER,BEFORE)==source
    ast.parse(effective)
    return effective


def exact_config_binding(workspace,config_path,pins):
    matches=[relative for relative in pins if relative==CONFIG_REL
             and (workspace/relative).resolve()==config_path]
    assert matches==[CONFIG_REL]
    assert sha(config_path)==pins[CONFIG_REL]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'), 'CPU audit requires hidden CUDA'
    assert args.cache_dir.resolve()==(CASE/'3d/cache').resolve()
    assert args.plan.resolve()==(SHADOW/'official_state_3d_cache_v1/producer_plan.json').resolve()
    assert args.output_dir.resolve()==(CASE/'3d/independent_audit_v2').resolve()
    assert sha(args.plan)==PLAN_SHA and sha(AUDITOR)==AUDITOR_SHA
    plan=json.loads(args.plan.read_text())
    workspace=Path(plan['workspace']).resolve()
    config=Path(plan['checkpoint']).resolve().parent/'config.json'
    exact_config_binding(workspace,config,plan['source_pins'])
    effective=patch(AUDITOR.read_text())
    metadata=CASE/'3d/audit_adapter_v2_metadata'
    metadata.mkdir(parents=True,exist_ok=False)
    (metadata/'effective_audit_source.py').write_text(effective,encoding='utf-8',newline='\n')
    launch=dict(actual_argv=sys.argv,started=time.time(),wrapper_sha256=sha(__file__),
        original_auditor_path=str(AUDITOR),original_auditor_sha256=AUDITOR_SHA,
        effective_audit_source_sha256=sha(metadata/'effective_audit_source.py'),
        plan_sha256=PLAN_SHA,config_relative_pin=CONFIG_REL,resolved_config=str(config),
        resolved_config_sha256=sha(config),original_file_modified=False,
        scope='One exact config membership guard; original feature SHA/finite/ID/stat/terminal checks unchanged',
        GPU_NN_render_encoding_or_old_raw_scan=False,old_failed_audit_preserved=True)
    with (metadata/'launch.json').open('x',encoding='utf-8') as stream:
        json.dump(launch,stream,indent=2,allow_nan=False)
        stream.write('\n')
    sys.path.insert(0,str(AUDITOR.parent))
    sys.argv=[str(AUDITOR),'--cache-dir',str(args.cache_dir),'--plan',str(args.plan),
              '--output-dir',str(args.output_dir)]
    namespace=dict(__name__='_exact_Double_config_path_auditor',__file__=str(AUDITOR))
    exec(compile(effective,str(AUDITOR),'exec'),namespace)
    namespace['main']()


if __name__=='__main__':
    main()
