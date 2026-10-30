"""Only install the new input adapter; reuse original GPU/CPU boundary functions."""
import argparse
import importlib
import json
from pathlib import Path
import traceback

from extension_runtime import install


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('gate','reload','accept'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--input-mode',choices=('baseline','prefetch1'),default='baseline')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    runtime=install(args.input_mode)
    boundary=importlib.import_module('gciql_boundary')
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True)
    try:
        if args.action=='gate':boundary.gate(config,args.output)
        elif args.action=='reload':boundary.reload_gate(config,args.output,args.report)
        else:boundary.accept(config,args.output,args.report)
    except BaseException:
        runtime.utility.atomic_json(args.report or args.output/'failure.json',dict(actual_exit_code=1,
            action=args.action,role=config['role'],error=traceback.format_exc()))
        raise
