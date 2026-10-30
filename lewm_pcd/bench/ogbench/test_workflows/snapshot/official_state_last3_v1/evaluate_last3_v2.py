"""Evaluation only for existing 800K/900K/1M models; never call main/update."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
FORMAL = CASE / 'official_state_formal_v1'
FORMAL_SHA = '616c2905f802065af10277674f39dde802075ecb9ab4ede40cc785d358a80c92'
STEPS = (800000, 900000, 1000000)
formal_source = (FORMAL / 'formal_main.py').read_bytes()
assert hashlib.sha256(formal_source).hexdigest() == FORMAL_SHA
_spec = importlib.util.spec_from_file_location('_last3_frozen_formal', FORMAL / 'formal_main.py')
formal = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = formal
_spec.loader.exec_module(formal)


def subset_method():
    root = ast.parse(formal_source)
    owner = next(node for node in root.body if isinstance(node, ast.ClassDef) and node.name == 'FormalInputBridge')
    original = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'run_deferred_evaluations')
    patched = copy.deepcopy(original)
    candidates = [node for node in ast.walk(patched) if isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'expected']
    assert len(candidates) == 1
    selected = candidates[0]
    old = copy.deepcopy(selected.value)
    selected.value = ast.List(elts=[ast.Constant(value=step) for step in STEPS], ctx=ast.Load())
    restored = copy.deepcopy(patched)
    restored_assignment = next(node for node in ast.walk(restored) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'expected')
    restored_assignment.value = old
    dump = lambda node: ast.dump(node, include_attributes=False)
    assert dump(restored) == dump(original), 'Unexpected method change outside node selection'
    namespace = dict(formal.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[patched], type_ignores=[])),
                 '<last3_single_expected_assignment>', 'exec'), namespace)
    proof = dict(formal_source_sha256=FORMAL_SHA, changed_AST_sites=1, all_other_method_AST_equal=True,
                 selected_checkpoint_steps=list(STEPS), change='expected node selection assignment only')
    return namespace['run_deferred_evaluations'], proof


_subset_run, SUBSET_PROOF = subset_method()


class LastThreeBridge(formal.FormalInputBridge):
    run_deferred_evaluations = _subset_run

    def mark(self, phase, **fields):
        record = dict(phase=phase, started=self.started, updated=time.time(),
            training_updates_in_this_process=0, source_training_updates=1000000,
            model_dir=str(self.official.FLAGS.save_dir), results_dir=str(self.state_dir),
            evaluation_checkpoint=self.eval_epoch, **fields)
        with (self.state_dir / 'entry_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        path = self.state_dir / 'entry_status.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(path)


def load_numpy(path):
    with np.load(path, allow_pickle=False) as data:
        return (str(data['generator'].item()), data['keys'].copy(), int(data['position']),
                int(data['has_gauss']), float(data['cached_gaussian']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--ledger-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    model, ledger_dir, out = args.model_dir.resolve(), args.ledger_dir.resolve(), args.output_dir.resolve()
    assert HERE in out.parents and out != model and out != ledger_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'eval.csv').exists(), 'Refuse to overwrite new or old evaluation results'
    old_csv = model / 'eval.csv'
    old_csv_before = formal.sha(old_csv) if old_csv.exists() else None
    formal.write_new(out / 'subset_AST_proof.json', SUBSET_PROOF)
    (out / 'entry_events.jsonl').open('x').close()
    source = CASE / 'source/ogbench'
    for name, expected in formal.PINS.items():
        assert formal.sha(source / name) == expected, name
    sys.path[:0] = [str(source / 'impls'), str(source)]
    spec = importlib.util.spec_from_file_location('_last3_official_main_globals', source / 'impls/main.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    # Parse the original flags so original log_utils.get_flag_dict also works.
    # Importing/compiling main does not call it; only its evaluation body is run.
    official.FLAGS(['evaluate_last3.py', '--seed=0', '--env_name=cube-triple-play-v0',
        '--run_group=Official_Triple_State_Prefetch1_Seed0_Last3', '--train_steps=1000000',
        '--eval_interval=100000', '--save_interval=100000', '--eval_episodes=50',
        '--eval_on_cpu=1', '--video_episodes=1', '--save_dir=' + str(model),
        '--agent=' + str(source / 'impls/agents/gciql.py')])
    config = official.FLAGS.agent
    config.alpha = 1.0
    recorded = json.loads((ledger_dir / 'entry_binding.json').read_text())['official_flags']['agent']
    assert json.loads(json.dumps(config.to_dict())) == recorded
    assert config['batch_size'] == 1024 and config['alpha'] == 1.0 and config['encoder'] is None
    assert config['frame_stack'] is None and config['discrete'] is False
    assert official.FLAGS.eval_gaussian is None and official.FLAGS.eval_temperature == 0
    import flax
    import ogbench
    from agents.gciql import GCIQLAgent
    assert official.jax.default_backend() == 'cpu', 'This evaluation-only entry requires CPU backend'
    env = ogbench.make_env_and_datasets('cube-triple-play-v0', env_only=True)
    env.reset()  # Match the original env-utils bootstrap, without loading data.
    assert len(env.unwrapped.task_infos) == 5 and env.spec.max_episode_steps == 1000
    original_logger, original_setup, original_evaluate = official.CsvLogger, official.setup_wandb, official.evaluate
    official.CsvLogger = lambda requested_path: original_logger(str(out / 'eval.csv'))

    def offline(*args, **kwargs):
        kwargs['mode'] = 'offline'
        return original_setup(*args, **kwargs)

    official.setup_wandb = offline
    bridge = LastThreeBridge(official, Path('/unused_no_dataset_loaded'), out)
    bridge.step = 1000000  # Historical source budget only; this process performs zero updates.

    def evaluate(*args, **kwargs):
        bridge.mark('evaluation', task_id=kwargs.get('task_id'), **bridge.counts(kwargs['agent']))
        result = original_evaluate(*args, **kwargs)
        bridge.mark('evaluation_task_returned', task_id=kwargs.get('task_id'))
        return result

    official.evaluate = evaluate
    original_restore = official.restore_agent

    def checked_restore(agent, restore_path, epoch, purpose='evaluation'):
        assert epoch in STEPS and Path(restore_path) == model
        path = model / f'params_{epoch}.pkl'
        sidecar = json.loads((model / f'checkpoint_{epoch}_sidecar.json').read_text())
        assert path.stat().st_size == sidecar['checkpoint_bytes']
        assert formal.sha(path) == sidecar['checkpoint_sha256']
        value = original_restore(agent, str(model), epoch)
        counts = bridge.counts(value)
        assert counts == dict(network_step=epoch + 1, adam_count=epoch)
        leaves = official.jax.tree_util.tree_leaves(flax.serialization.to_state_dict(value))
        assert len(leaves) == 153 and all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
        formal.write_new(out / f'restore_{purpose}_{epoch}.json', dict(
            checkpoint_epoch=epoch, checkpoint_path=str(path), checkpoint_sha256=sidecar['checkpoint_sha256'],
            actual_exit_code=0, state_leaves=153, all_state_finite=True, **counts,
            training_updates_in_this_process=0))
        return value

    official.restore_agent = checked_restore
    template = GCIQLAgent.create(0, np.zeros((1, 46), np.float32), np.zeros((1, 5), np.float32), config)
    bridge.final_agent = checked_restore(template, model, 1000000, purpose='template')
    ledger_pointers = {}
    for step in STEPS:
        path = ledger_dir / f'eval_node_{step}_seed_ledger.json'
        record = json.loads(path.read_text())
        before_path = ledger_dir / f'eval_node_{step}_before_numpy.npz'
        after_path = ledger_dir / f'eval_node_{step}_after_numpy.npz'
        assert record['step'] == step and [item['task_id'] for item in record['task_seeds']] == [1, 2, 3, 4, 5]
        assert formal.sha(before_path) == record['before_numpy_sha256']
        assert formal.sha(after_path) == record['after_numpy_sha256']
        bridge.nodes.append(dict(step=step, before=load_numpy(before_path), after=load_numpy(after_path),
                                 env=env, config=config))
        ledger_pointers[str(step)] = dict(path=str(path), sha256=formal.sha(path), task_seeds=record['task_seeds'])
    code, proof = formal.compile_deferred_main(official, (source / 'impls/main.py').read_bytes(), lambda *a: None)
    formal.write_new(out / 'original_eval_body_binding.json', proof)
    formal.write_new(out / 'entry_binding.json', dict(entry_sha256=formal.sha(__file__),
        frozen_formal_sha256=FORMAL_SHA, checkpoint_steps=list(STEPS), model_dir=str(model),
        source_training_updates=1000000, training_updates_in_this_process=0,
        original_main_called=False, agent_update_called=False, dataset_loaded=False,
        evaluation=dict(tasks=5, episodes_per_task=50, video_episodes_per_task=1,
                        max_episode_steps=1000, temperature=0., Gaussian=None, CPU=True),
        original_protocol_and_actor_seeds_preserved=True, same_initial_scenes_as_stopped_run=False,
        environment_rng='new independent Gym Generator; prior live state was not saved',
        ledger_pointers=ledger_pointers, official_flags=official.get_flag_dict(),
        eval_csv=str(out / 'eval.csv'), old_eval_csv_target_redirected=True,
        old_eval_csv_before_sha256=old_csv_before, wandb_mode='offline'))
    bridge.mark('last3_evaluation_started')
    try:
        bridge.run_deferred_evaluations(code)
        old_csv_after = formal.sha(old_csv) if old_csv.exists() else None
        assert old_csv_after == old_csv_before, 'Old eval.csv changed during new evaluation'
        formal.write_new(out / 'old_eval_csv_protected.json', dict(actual_exit_code=0,
            path=str(old_csv), before_sha256=old_csv_before, after_sha256=old_csv_after))
        bridge.mark('last3_evaluation_returned', actual_exit_code=0)
    except BaseException as exc:
        bridge.mark('last3_evaluation_failed', actual_exit_code=1, error=repr(exc))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
