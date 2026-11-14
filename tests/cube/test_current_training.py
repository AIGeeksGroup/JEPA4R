# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/tests/test_current_training.py (verbatim except: imports /
# paths adapted to the package; lines wrapped at 120 columns, no lint exemptions, unused imports removed)
"""Current paper training and source-signature contracts."""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

LEWM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LEWM))


# the packages the split placed them in
PACKAGES = {'fresh_data': 'jepa4r.data.cube', 'fresh_train': 'jepa4r.models.cube',
            'training_runtime': 'jepa4r.models.cube', 'pldm.shared_data': 'jepa4r.baselines.cube'}


def api(name):
    prefix = PACKAGES.get(name, 'jepa4r.policy.gciql')
    module = prefix + '.' + name
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, 'clean current training projection is missing'
    return importlib.import_module(module)


def bind(tmp_path, cubes):
    data = api('fresh_data')
    horizon = {1: 200, 2: 500, 3: 1000, 4: 600}[cubes]
    producer = tmp_path / 'producer.py'
    producer.write_text('# current fixture producer\n')
    value = dict(schema='cube_fixed_collection_v1', cube_count=cubes, mode='smoke', raw_horizon=horizon,
                 train_episodes=2, validation_episodes=1, split_counts={'train': 2, 'validation': 1},
                 actual_exit_code=0, paths={}, protocol='paper_cube_fixed_mix_v1', collection_seed=0,
                 collection_protocol={'name': 'paper_cube_fixed_mix_v1'},
                 producer_binding={'files': {str(producer): data.sha256(producer)}})
    for split, episodes in [('train', 2), ('validation', 1)]:
        rows = episodes * (horizon + 1)
        action = np.zeros((rows, 5), np.float32)
        action[np.arange(episodes) * (horizon + 1) + horizon] = np.nan
        for mode in ('rgb', 'pcd'):
            path = tmp_path / f'{split}_{mode}.h5'
            with h5py.File(path, 'x', libver='latest') as f:
                f['action'] = action
                f['ep_len'] = np.full(episodes, horizon + 1, np.int32)
                f['ep_offset'] = np.arange(episodes, dtype=np.int64) * (horizon + 1)
                for key, width in [('observation', 19 + 9 * cubes), ('qpos', 14 + 7 * cubes), ('qvel', 14 + 6 * cubes)]:
                    f.create_dataset(key, shape=(rows, width), dtype='f8')
                if mode == 'rgb':
                    f.create_dataset('pixels', shape=(rows, 224, 224, 3), dtype='u1')
                else:
                    f.create_dataset('pcd_xyz', shape=(rows, 1200, 3), dtype='f4')
                    f.create_dataset('pcd_rgb', shape=(rows, 1200, 3), dtype='u1')
            value['paths'][split + '_' + mode] = str(path)
    collected = tmp_path / 'COLLECTION.json'
    collected.write_text(json.dumps(value))
    return data.create_binding(collected, tmp_path / 'data')


@pytest.mark.parametrize('cubes,horizon', [(1, 200), (2, 500), (3, 1000), (4, 600)])
def test_all_cubes_use_all_action_rows_and_the_same_heldout_seed0_recipe(tmp_path, cubes, horizon):
    manifest = bind(tmp_path, cubes)
    data, train = api('fresh_data'), api('fresh_train')
    bound = data.load_binding(manifest)
    assert bound['training_rows_per_episode'] == horizon
    assert bound['training_seed'] == 0
    for row in bound['views'].values():
        with h5py.File(row['path'], 'r') as f:
            assert np.all(f['ep_len'][:] == horizon)
            assert np.isfinite(f['action'][:]).all()
    for method in ('lewm_rgb', 'lewm_pcd'):
        plan = train.build_plan(cubes, method, manifest, tmp_path / method, mode='smoke')
        cfg = plan['config']
        assert cfg['seed'] == cfg['fresh_seed'] == 0
        assert 'train_split' not in cfg
        assert cfg['loader']['batch_size'] == 128
        assert cfg['trainer']['precision'] == 'bf16-mixed'
        assert cfg['optimizer'] == {'type': 'AdamW', 'lr': 5e-5, 'weight_decay': .001}
        assert cfg['loss']['sigreg']['weight'] == .09
        assert cfg['data']['dataset']['num_steps'] == 4
        assert cfg['data']['dataset']['frameskip'] == 5
        assert cfg['data']['dataset']['name'] != cfg['data']['validation_dataset']['name']
        assert plan['protocol'] == 'episode_heldout_v1'
        assert not (tmp_path / method).exists()


def test_train_modes_and_cube_counts_are_explicit(tmp_path):
    manifest = bind(tmp_path, 1)
    train = api('fresh_train')
    with pytest.raises(ValueError, match='mode'):
        train.build_plan(1, 'pldm', manifest, tmp_path / 'new', mode='formal')
    with pytest.raises(ValueError, match='Cube'):
        train.build_plan(2, 'pldm', manifest, tmp_path / 'new', mode='smoke')
    tiny = train.build_plan(1, 'lejepa', manifest, tmp_path / 'tiny', mode='smoke')
    assert tiny['recipe']['epochs'] == 10 and tiny['mode'] == 'smoke'
    pldm = api('pldm.shared_data')
    assert pldm.contract('cube1', data_binding=manifest)['train_samples'] == 2 * 185
    with pytest.raises(ValueError):
        pldm.contract('cube1')


@pytest.mark.parametrize(
    'field,value', [('protocol', 'other'), ('collection_seed', 3072), ('collection_protocol', {'name': 'other'})]
)
def test_binding_rejects_noncurrent_collection_contract(tmp_path, field, value):
    bind(tmp_path, 1)
    data = api('fresh_data')
    source = tmp_path / 'COLLECTION.json'
    collected = json.loads(source.read_text())
    collected[field] = value
    source.write_text(json.dumps(collected))
    with pytest.raises(ValueError, match='protocol|seed'):
        data.create_binding(source, tmp_path / 'rejected')




def _gate_none_condition(node):
    if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and node.left.id == 'gate' and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None):
        if isinstance(node.ops[0], ast.Is):
            return True
        if isinstance(node.ops[0], ast.IsNot):
            return False
    # Only a false first operand may discard an otherwise unknown call: Python short-circuits it.
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        if _gate_none_condition(node.values[0]) is False:
            return False
    return None


class _TinyGateNone(ast.NodeTransformer):
    """Select only statically unreachable gate branches; retain every ordinary computation."""
    def __init__(self):
        self.zero_starts = set()

    def visit_If(self, node):
        condition = _gate_none_condition(node.test)
        if condition is None:
            return self.generic_visit(node)
        branch = ast.Module(body=node.body if condition else node.orelse, type_ignores=[])
        return self.visit(branch).body

    def visit_IfExp(self, node):
        condition = _gate_none_condition(node.test)
        return self.generic_visit(node) if condition is None else self.visit(node.body if condition else node.orelse)

    def visit_Assign(self, node):
        zero_start = (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ('start_epoch', 'start_batch')
                      and isinstance(node.value, ast.IfExp) and _gate_none_condition(node.value.test) is True
                      and isinstance(node.value.body, ast.Constant) and type(node.value.body.value) is int
                      and node.value.body.value == 0)
        node = self.generic_visit(node)
        if zero_start:
            self.zero_starts.add(node.targets[0].id)
            return None
        return node

    def visit_For(self, node):
        # Capture bindings before visiting the body: a later assignment cannot justify the loop's start.
        zero_starts = self.zero_starts.copy()
        node = self.generic_visit(node)
        call = node.iter
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            if (call.func.id == 'range' and len(call.args) == 2 and not call.keywords
                    and isinstance(call.args[0], ast.Name) and call.args[0].id == 'start_epoch'
                    and 'start_epoch' in zero_starts):
                call.args = call.args[1:]
            if (call.func.id == 'enumerate' and len(call.args) == len(call.keywords) == 1
                    and call.keywords[0].arg == 'start' and isinstance(call.keywords[0].value, ast.Name)
                    and call.keywords[0].value.id == 'start_batch' and 'start_batch' in zero_starts):
                call.keywords = []
        return node


def _tiny_default_off_ast_sha256(text):
    main = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    args = main.args
    if (args.posonlyargs or args.args or args.vararg or args.kwarg or args.defaults
            or main.decorator_list or main.returns is not None):
        raise ValueError('Tiny main must retain its original zero-positional-argument contract')
    if args.kwonlyargs:
        if ([arg.arg for arg in args.kwonlyargs] != ['gate'] or len(args.kw_defaults) != 1
                or args.kwonlyargs[0].annotation is not None or not isinstance(args.kw_defaults[0], ast.Constant)
                or args.kw_defaults[0].value is not None):
            raise ValueError('Tiny instrumentation must default to gate=None')
        if any(isinstance(node, ast.Name) and node.id == 'gate' and not isinstance(node.ctx, ast.Load)
               for node in ast.walk(main)):
            raise ValueError('Tiny gate cannot be rebound')
        body = _TinyGateNone().visit(ast.Module(body=main.body, type_ignores=[]))
    else:
        body = ast.Module(body=main.body, type_ignores=[])
    if any(isinstance(node, ast.Name) and node.id == 'gate' for node in ast.walk(body)):
        raise ValueError('unhandled gate-dependent computation remains')

    def shape(value):
        if isinstance(value, ast.AST):
            return [type(value).__name__, [[name, shape(child)] for name, child in ast.iter_fields(value)]]
        if isinstance(value, list):
            return [shape(child) for child in value]
        return value
    # Explicit fields avoid Python-version-dependent ast.dump omission of empty fields.
    return hashlib.sha256(json.dumps(shape(body), separators=(',', ':')).encode()).hexdigest()


def test_projected_algorithms_match_original_function_bytes():
    package = LEWM / 'jepa4r/policy/gciql'
    signatures = json.loads((Path(__file__).parent / 'ALGORITHM_SIGNATURES.json').read_text())
    for record in signatures['functions']:
        text = (package / record['file']).read_text(encoding='utf8')
        tree = ast.parse(text)
        node = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == record['name']
        )
        actual = hashlib.sha256(ast.get_source_segment(text, node).encode()).hexdigest()
        assert actual == record['sha256'], (record['file'], record['name'])
    reference = signatures['tiny_default_off_main_ast_v1']
    assert reference['original_optimization_block_sha256'] == signatures['tiny_optimization_block_sha256']
    text = (LEWM / 'jepa4r/baselines/cube/native/tiny_train.py').read_text()
    assert _tiny_default_off_ast_sha256(text) == reference['sha256']


@pytest.mark.parametrize(
    'before,after',
    [
        ('torch.manual_seed(0)', 'torch.manual_seed(1)'),
        ('drop_last=True', 'drop_last=False'),
        ("'weight_decay': 5e-2", "'weight_decay': 5e-3"),
        ('start_factor=0.01', 'start_factor=0.02'),
        ('eta_min=1e-3', 'eta_min=1e-2'),
        ('GradScaler(enabled=True)', 'GradScaler(enabled=False)'),
        ('dtype=torch.bfloat16', 'dtype=torch.float32'),
        ('sigreg_loss * args.lamb + inv_loss', 'sigreg_loss * args.lamb - inv_loss'),
        (
            '            scaler.update()\n            scheduler.step()',
            '            scheduler.step()\n            scaler.update()',
        ),
        ('start_epoch = 0 if gate is None', 'start_epoch = 1 if gate is None'),
        ('start_batch = 0 if gate is None', 'start_batch = 1 if gate is None'),
    ],
)
def test_tiny_default_off_signature_rejects_real_training_changes(before, after):
    text = (LEWM / 'jepa4r/baselines/cube/native/tiny_train.py').read_text()
    assert text.count(before) == 1
    signatures = json.loads((Path(__file__).parent / 'ALGORITHM_SIGNATURES.json').read_text())
    expected = signatures['tiny_default_off_main_ast_v1']['sha256']
    assert _tiny_default_off_ast_sha256(text.replace(before, after)) != expected


@pytest.mark.parametrize('before,after', [
    ('def main(*, gate=None)', 'def main(*, gate=True)'),
    ('    torch.manual_seed(0)', '    gate = object()\n    torch.manual_seed(0)'),
    ('gate is not None and gate.after_step', 'gate is not None or gate.after_step'),
    ('            if gate is not None:\n                gate.before_step(net)', '            gate.before_step(net)'),
])
def test_tiny_default_off_signature_rejects_live_or_unguarded_hooks(before, after):
    text = (LEWM / 'jepa4r/baselines/cube/native/tiny_train.py').read_text()
    assert text.count(before) == 1
    with pytest.raises(ValueError, match='gate'):
        _tiny_default_off_ast_sha256(text.replace(before, after))


def test_current_runtime_source_manifest_matches_exact_files():
    runtime = api('training_runtime')
    assert runtime.verify_sources()['mismatches'] == []


