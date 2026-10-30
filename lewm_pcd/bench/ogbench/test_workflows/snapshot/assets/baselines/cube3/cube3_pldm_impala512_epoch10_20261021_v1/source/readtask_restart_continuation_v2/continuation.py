"""Resume the verified complete epoch3; reuse the frozen read-task continuation."""
import ast
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT_UPDATES, TOTAL_UPDATES = 137109, 457030
LEGACY_SHA = 'e9768305dded55793c9e3cd8e344f71a2439390c2513da7b335e5bc2837f7250'
_API = None


def epoch3_tree(source):
    counts = [0] * 7
    class Patch(ast.NodeTransformer):
        active = None
        def visit_FunctionDef(self, node):
            if self.active is not None:
                return self.generic_visit(node)
            if node.name not in ('protocol', 'run_train', 'acceptance_tree', 'run_accept', 'main'):
                return node
            self.active = node.name
            node = self.generic_visit(node)
            self.active = None
            return node
        def visit_Compare(self, node):
            text = ast.unparse(node)
            if self.active == 'run_train' and text == 'trainer.current_epoch != 2':
                counts[0] += 1
                node.comparators = [ast.Constant(3)]
            return self.generic_visit(node)
        def visit_keyword(self, node):
            if self.active == 'run_train' and node.arg == 'epoch_index' and isinstance(node.value, ast.Constant) and node.value.value == 2:
                counts[1] += 1
                node.value = ast.Constant(3)
            if self.active == 'run_accept' and node.arg in ('inherited_complete_validation_epochs', 'new_complete_validation_epochs'):
                expected = 2 if node.arg.startswith('inherited') else 8
                if not isinstance(node.value, ast.Constant) or node.value.value != expected:
                    raise ValueError('native acceptance epoch metadata boundary changed')
                counts[2] += 1
                node.value = ast.Constant(3 if expected == 2 else 7)
            if self.active == 'main' and node.arg == 'new_full_validation_epochs':
                if ast.unparse(node.value) != '8': raise ValueError('prepare budget boundary changed')
                counts[3] += 1
                node.value = ast.Constant(7)
            return self.generic_visit(node)
        def visit_BinOp(self, node):
            if self.active in ('protocol', 'run_accept') and ast.unparse(node) == '8 * 4571':
                counts[4] += 1
                node.left = ast.Constant(7)
            return self.generic_visit(node)
        def visit_Constant(self, node):
            if self.active == 'acceptance_tree' and isinstance(node.value, str):
                value = node.value.replace('epochs[2:]', 'epochs[3:]').replace('paths[2:]', 'paths[3:]')
                if value != node.value:
                    counts[5] += 1
                    return ast.copy_location(ast.Constant(value), node)
            return node
        def visit_Call(self, node):
            if self.active == 'acceptance_tree' and ast.unparse(node.func) == 'ast.Constant' and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 8:
                counts[6] += 1
                node.args[0] = ast.Constant(7)
            return self.generic_visit(node)
    tree = Patch().visit(ast.parse(source))
    if counts != [1, 1, 2, 1, 2, 2, 1]:
        raise ValueError('frozen epoch3 adaptation boundaries changed: '+str(counts))
    return ast.fix_missing_locations(tree)


def api():
    global _API
    if _API is None:
        import hashlib
        path = HERE.parent/'readtask_continuation_v1/continuation.py'
        if hashlib.sha256(path.read_bytes()).hexdigest() != LEGACY_SHA:
            raise ValueError('frozen original continuation changed')
        spec = importlib.util.spec_from_file_location('restart_frozen_continuation', path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = value
        exec(compile(epoch3_tree(path.read_text()), str(path), 'exec'), value.__dict__)
        value.HERE = HERE
        value.PARENT_UPDATES = PARENT_UPDATES
        value.require_runtime = require_runtime
        value.lineage = lineage
        _API = value
    return _API


def lineage(rows, parents):
    if len(parents) != 3 or [r['epoch'] for r in rows] != list(range(1, 11)):
        raise ValueError('three inherited plus seven produced epoch exports required')
    if any((rows[i]['path'], rows[i]['sha256']) != (parents[i]['path'], parents[i]['sha256']) for i in range(3)):
        raise ValueError('actual inherited epoch1..3 exports differ')
    return [dict(row, origin=parents[i]['origin'] if i < 3 else 'produced_by_this_run') for i, row in enumerate(rows)]


def require_runtime(a):
    m = api()
    old = m.base()
    trial, data_root, freeze = old.source_guard(a.root)
    output = a.output if a.mode == 'full' else a.output.parents[2]
    release = m.read(a.release)
    expected = dict(stage='approved_pldm_readtask_restart_continuation', output=str(output),
        source_freeze_sha256=m.sha(HERE/'source_freeze_v1.json'), parent_proof=str(a.parent_proof),
        parent_proof_sha256=m.sha(a.parent_proof), trial_report=str(a.trial_report),
        trial_report_sha256=m.sha(a.trial_report))
    if any(release.get(k) != v for k, v in expected.items()):
        raise ValueError('matching separate Root restart continuation release required')
    probe = old.load('restart_closed_trial', a.root/'readtask_restart_probe_v2/trial.py')
    probe.verify_interruption(release['interruption_receipt'], release['interruption_receipt_sha256'],
                              check_live=a.mode in ('full', 'train'))
    report = probe.verify_closed_trial(a.trial_report, a.parent_proof)
    parent = probe.validate_parent(m.read(a.parent_proof))
    profile = report['selected_profile']
    m.protocol(profile)
    if release.get('selected_profile') != profile:
        raise ValueError('Root profile differs from actual verified comparison')
    refs = parent['prior_parent_references']
    for index, key in enumerate(('P1', 'P2')):
        ref = refs[key]
        if m.sha(ref['proof_path']) != ref['proof_sha256']:
            raise ValueError('accepted prior epoch proof changed')
        row = parent['epoch_exports'][index]
        bare = m.read(ref['proof_path'])['files']['bare']
        if (row['path'], row['sha256']) != (bare['path'], bare['sha256']):
            raise ValueError('inherited epoch export provenance changed')
        stat = Path(bare['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (bare['bytes'], bare['mtime_ns']):
            raise ValueError('inherited epoch bare stat changed')
    prior = parent['baseline_binding']['continuation']
    if prior.get('source_freeze_sha256') != m.F2B_FREEZE or prior.get('parent_proof_sha256') != refs['P1']['proof_sha256']:
        raise ValueError('actual epoch3 producer binding changed')
    bare = parent['files']['bare']
    if (parent['epoch_exports'][2]['path'], parent['epoch_exports'][2]['sha256']) != (bare['path'], bare['sha256']):
        raise ValueError('protected latest epoch3 export differs')
    for item in parent['files'].values():
        stat = Path(item['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise ValueError('protected latest parent stat changed')
    from serial_relay import require_prerequisite
    if require_prerequisite(a.root/'prerequisites/triple_complete_ready.json', data_root) is None:
        raise ValueError('original accepted Triple prerequisite absent')
    if a.mode == 'accept' and a.training_output is None:
        raise ValueError('explicit own training output required')
    m._PROFILE = profile
    return trial, data_root, parent, profile, freeze


if __name__ == '__main__':
    raise SystemExit(api().main())
