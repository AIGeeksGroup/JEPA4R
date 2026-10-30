"""Thin entry: selected prefetch1, explicit deferred eval and first-step save."""
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
OLD = CASE / 'official_state_workers_v1'
MAIN_SHA = '375fad6375047ff59f3baea408ab7197784c1ec1457588a78a631025e434faee'
ADAPTER_SHA = '4508ce847694976f8af924215d11985ac4d795093b22d62e8424928346fbdda2'
SELECTED_SHA = 'bdcfec0a47eef5c4913505207c743a655e1afa423f37ad51768f611228e68471'
PINS = {'impls/main.py': MAIN_SHA,
    'impls/agents/gciql.py': '05a0b5d9b987d3049df8a1158bbfbaba2ca2611604d24f9e484f8fb6d0cd92bc',
    'impls/utils/datasets.py': '52c373922a87e1ba8ad8b3b96c3474d5e6385e2ea2c7206930f7a8a315816609',
    'impls/utils/env_utils.py': 'a92d391a3d95e26717b80ea783f487f0639c956f23ef603bfa8b7bb098f763d7',
    'impls/utils/evaluation.py': '4d9a2732bc9ef89ce99fbc0f07a76d3f4d035783af3a658251709140ff9de3d5',
    'impls/utils/log_utils.py': 'ba5ad408d91ee35d92bf06ec7846846c5a75af4ae7cc62c329ec43d074bfaeb6',
    'impls/utils/flax_utils.py': '5b6a8e9a919bf57dae0c1454f327ded4235385f9d9dcd1e62263fe4fad9fd484',
    'impls/hyperparameters.sh': '633f6e1f5311e42d0bfd6524cda7255120a0835678dcaa1227cfbf3f5cb1d9df',
    'ogbench/utils.py': 'f503a3a0497f91ce02958abae5770f3e08c9d6a3a740687c828c996fedb24e9b'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def ordered_input_class():
    path = OLD / 'ordered_input.py'
    assert sha(path) == ADAPTER_SHA
    spec = importlib.util.spec_from_file_location('_formal_frozen_ordered_input', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.OrderedInput


def compile_deferred_main(official, source, callback):
    """Change exactly two AST sites in memory; preserve source bytes on disk."""
    assert hashlib.sha256(source).hexdigest() == MAIN_SHA
    original = ast.parse(source)
    patched = copy.deepcopy(original)
    main = next(node for node in patched.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loop = [node for node in main.body if isinstance(node, ast.For)]
    assert len(loop) == 1
    dump = lambda node: ast.dump(node, include_attributes=False)
    eval_test = dump(ast.parse('i == 1 or i % FLAGS.eval_interval == 0', mode='eval').body)
    save_test = dump(ast.parse('i % FLAGS.save_interval == 0', mode='eval').body)
    eval_nodes = [node for node in loop[0].body if isinstance(node, ast.If) and dump(node.test) == eval_test]
    save_nodes = [node for node in loop[0].body if isinstance(node, ast.If) and dump(node.test) == save_test]
    assert len(eval_nodes) == len(save_nodes) == 1
    evaluation, saving = eval_nodes[0], save_nodes[0]
    original_body, original_save_test = copy.deepcopy(evaluation.body), copy.deepcopy(saving.test)
    eval_code = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(original_body), type_ignores=[])),
                        '<original_official_evaluation_body>', 'exec')
    evaluation.body = ast.parse('_defer_evaluation_node(agent, env, config, i)').body
    saving.test = ast.parse('i == 1 or i % FLAGS.save_interval == 0', mode='eval').body
    # Restore only the two declared sites, then compare the entire source AST.
    restored = copy.deepcopy(patched)
    restored_main = next(node for node in restored.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    restored_loop = next(node for node in restored_main.body if isinstance(node, ast.For))
    restored_loop.body[loop[0].body.index(evaluation)].body = original_body
    restored_loop.body[loop[0].body.index(saving)].test = original_save_test
    assert dump(restored) == dump(original), 'Unexpected AST modification outside declared sites'
    official.__dict__['_defer_evaluation_node'] = callback
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])),
                 '<official_main_two_explicit_extensions>', 'exec'), official.__dict__)
    proof = dict(original_source_sha256=MAIN_SHA, changed_AST_sites=2, all_other_AST_equal=True,
        changes=['replace only Evaluate-if body with seed-ledger callback', 'add first-step save guard'],
        original_evaluation_body_AST_sha256=hashlib.sha256(
            dump(ast.Module(body=original_body, type_ignores=[])).encode()).hexdigest())
    return eval_code, proof


class FormalInputBridge:
    """Patch only main's input/external-RNG boundaries, logging and data path."""
    def __init__(self, official, data_dir, state_dir):
        self.official, self.data_dir, self.state_dir = official, Path(data_dir), Path(state_dir)
        self.step = 0
        self.inputs = None
        self.datasets = []
        self.original_samples = []
        self.started = time.time()
        self.nodes = []
        self.final_agent = None
        self.eval_epoch = None

    def mark(self, phase, **fields):
        record = dict(phase=phase, started=self.started, updated=time.time(),
            training_batches_returned=self.step, actual_save_dir=str(self.official.FLAGS.save_dir),
            pending=bool(self.inputs and self.inputs.pending), evaluation_checkpoint=self.eval_epoch, **fields)
        with (self.state_dir / 'entry_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        target = self.state_dir / 'entry_status.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(target)

    def barrier(self):
        if self.inputs is not None:
            self.inputs.barrier()

    def counts(self, agent):
        return dict(network_step=int(agent.network.step), adam_count=int(agent.network.opt_state[0].count))

    def save_numpy(self, path, state):
        with Path(path).open('xb') as stream:
            np.savez(stream, generator=np.array(state[0]), keys=state[1], position=np.array(state[2]),
                     has_gauss=np.array(state[3]), cached_gaussian=np.array(state[4]))

    def defer_evaluation_node(self, agent, env, config, step):
        self.barrier()
        assert self.step == step and config.get('discrete') is False
        assert self.official.FLAGS.eval_gaussian is None
        infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
        count = self.official.FLAGS.eval_tasks or len(infos)
        before = np.random.get_state()
        seeds = [int(np.random.randint(0, 2**32)) for _ in range(count)]
        after = np.random.get_state()
        before_path = self.state_dir / f'eval_node_{step}_before_numpy.npz'
        after_path = self.state_dir / f'eval_node_{step}_after_numpy.npz'
        self.save_numpy(before_path, before)
        self.save_numpy(after_path, after)
        record = dict(step=step, task_seeds=[dict(task_id=i + 1, seed=seed) for i, seed in enumerate(seeds)],
            before_numpy_path=str(before_path), after_numpy_path=str(after_path),
            before_numpy_sha256=sha(before_path), after_numpy_sha256=sha(after_path),
            rollout_executed=False, SR_measured=False, **self.counts(agent))
        write_new(self.state_dir / f'eval_node_{step}_seed_ledger.json', record)
        self.nodes.append(dict(step=step, before=before, after=after, env=env, config=config))
        self.mark('evaluation_deferred_seed_ledger_saved', **record)

    def run_deferred_evaluations(self, eval_code):
        self.barrier()
        flags = self.official.FLAGS
        expected = [1] + list(range(flags.eval_interval, flags.train_steps + 1, flags.eval_interval))
        assert [node['step'] for node in self.nodes] == expected
        assert self.step == flags.train_steps and self.final_agent is not None
        train_tail = np.random.get_state()
        # A new offline run accepts checkpoint steps 1..1M after the training
        # run has already logged step1M; W&B rejects backwards steps in one run.
        self.official.wandb.finish()
        self.official.setup_wandb(project='OGBench', group=flags.run_group + '_deferred_eval',
                                  name=self.official.get_exp_name(flags.seed) + '_deferred_eval')
        logger = self.official.CsvLogger(str(Path(flags.save_dir) / 'eval.csv'))
        try:
            for node in self.nodes:
                self.eval_epoch = node['step']
                restored = self.official.restore_agent(self.final_agent, flags.save_dir, node['step'])
                assert self.counts(restored) == dict(network_step=node['step'] + 1, adam_count=node['step'])
                np.random.set_state(node['before'])
                namespace = dict(self.official.__dict__, agent=restored, env=node['env'],
                                 config=node['config'], eval_logger=logger, i=node['step'])
                # One namespace is required for comprehensions inside exec.
                exec(eval_code, namespace, namespace)
                actual, expected_state = np.random.get_state(), node['after']
                assert actual[0] == expected_state[0] and actual[2:] == expected_state[2:]
                assert np.array_equal(actual[1], expected_state[1]), 'Deferred evaluation RNG ledger mismatch'
                metrics = {key: float(value) for key, value in namespace['eval_metrics'].items() if key != 'video'}
                write_new(self.state_dir / f'deferred_eval_{node["step"]}.json',
                          dict(checkpoint_epoch=node['step'], metrics=metrics, actual_exit_code=0,
                               original_evaluation_body_used=True, RNG_ledger_match=True))
                self.mark('deferred_checkpoint_evaluation_returned', checkpoint_epoch=node['step'])
        finally:
            logger.close()
            self.official.wandb.finish()
            np.random.set_state(train_tail)
            self.eval_epoch = None

    def __enter__(self):
        module = self.official
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / 'entry_events.jsonl').open('x').close()
        names = ('GCDataset', 'make_env_and_datasets', 'setup_wandb', 'evaluate', 'save_agent')
        self.original = {name: getattr(module, name) for name in names}
        Input = ordered_input_class()

        def dataset(*args, **kwargs):
            value = self.original['GCDataset'](*args, **kwargs)
            original_sample = value.sample
            self.datasets.append(value)
            self.original_samples.append(original_sample)
            if self.inputs is None:
                # Copy into OrderedInput BEFORE attaching an instance method;
                # its deferred shallow copy must retain the original sampler.
                self.inputs = Input(value, mode='prefetch1', batch_size=1024)
                example_done = False

                def train_sample(batch_size, *args, **kwargs):
                    nonlocal example_done
                    if not example_done and batch_size == 1 and self.step == 0:
                        example_done = True
                        batch = original_sample(batch_size, *args, **kwargs)
                        self.mark('original_example_sample', example_batch_size=1)
                        return batch
                    assert example_done and batch_size == 1024 and not args and not kwargs
                    next_step = self.step + 1
                    flags = module.FLAGS
                    boundary = next_step == 1 or next_step == flags.train_steps or any(
                        next_step % interval == 0 for interval in
                        (flags.log_interval, flags.eval_interval, flags.save_interval))
                    batch = self.inputs.sample(prefetch_next=not boundary)
                    self.step = next_step
                    if boundary:
                        assert not self.inputs.pending
                        self.mark('boundary_batch_ready_for_update')
                    return batch

                value.sample = train_sample
            else:
                def val_sample(*args, **kwargs):
                    self.barrier()
                    self.mark('validation')
                    return original_sample(*args, **kwargs)
                value.sample = val_sample
            return value

        def make_env(*args, **kwargs):
            import ogbench
            original = ogbench.make_env_and_datasets

            def local_data(*args, **kwargs):
                assert not {'dataset_path', 'dataset_dir'} & kwargs.keys()
                # The original public API bypasses download_datasets when an
                # explicit accepted train path is supplied; val uses its suffix.
                return original(*args, dataset_path=str(self.data_dir / 'cube-triple-play-v0.npz'), **kwargs)

            ogbench.make_env_and_datasets = local_data
            try:
                return self.original['make_env_and_datasets'](*args, **kwargs)
            finally:
                ogbench.make_env_and_datasets = original

        def offline(*args, **kwargs):
            kwargs['mode'] = 'offline'
            return self.original['setup_wandb'](*args, **kwargs)

        def evaluate(*args, **kwargs):
            self.barrier()
            self.mark('evaluation', task_id=kwargs.get('task_id'), **self.counts(kwargs['agent']))
            result = self.original['evaluate'](*args, **kwargs)
            self.mark('evaluation_task_returned', task_id=kwargs.get('task_id'))
            return result

        def save(agent, save_dir, epoch):
            self.barrier()
            path = Path(save_dir) / f'params_{epoch}.pkl'
            assert not path.exists(), 'Refuse to overwrite a checkpoint'
            self.mark('saving', **self.counts(agent))
            result = self.original['save_agent'](agent, save_dir, epoch)
            state = np.random.get_state()
            rng_path = Path(save_dir) / f'numpy_rng_{epoch}.npz'
            self.save_numpy(rng_path, state)
            counts = self.counts(agent)
            assert counts == dict(network_step=epoch + 1, adam_count=epoch)
            pointer = dict(epoch=epoch, **counts, checkpoint_path=str(path), checkpoint_bytes=path.stat().st_size,
                checkpoint_sha256=sha(path), host_numpy_rng_sidecar=str(rng_path),
                host_numpy_rng_sidecar_sha256=sha(rng_path), host_numpy_rng_in_original_pkl=False)
            write_new(Path(save_dir) / f'checkpoint_{epoch}_sidecar.json', pointer)
            self.mark('checkpoint_saved', **pointer)
            self.final_agent = agent
            return result

        for name, value in dict(GCDataset=dataset, make_env_and_datasets=make_env,
                                setup_wandb=offline, evaluate=evaluate, save_agent=save).items():
            setattr(module, name, value)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self.inputs is not None:
                self.inputs.close()
            self.mark('official_main_returned' if exc_type is None else 'entry_failed',
                      actual_exit_code=0 if exc_type is None else 1, error=None if exc is None else repr(exc))
        finally:
            for name, original in self.original.items():
                setattr(self.official, name, original)
            for value, sample in zip(self.datasets, self.original_samples):
                value.sample = sample


def verify_data():
    acceptance = OLD / 'data_prepare/ACCEPTANCE.json'
    freeze_path = OLD / 'data_prepare/stat-freeze.json'
    frozen = json.loads(freeze_path.read_text())
    assert sha(acceptance) == '916c1921cfe8200b0a7c9dc78ee1b846626a2f4ff5867d757d73b50797fa680e'
    assert sha(freeze_path) == '72b1e240017d4ee56e147b3a2554ea33dda2d6820d7886922cd2db9de9d34f7f'
    assert frozen['actual_exit_code'] == 0 and frozen['acceptance_sha256'] == sha(acceptance)
    for item in frozen['datasets'].values():
        path = Path(item['path'])
        assert path.parent == CASE / 'data/official_state'
        st = path.stat()
        actual = dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns,
                      device=st.st_dev, inode=st.st_ino)
        assert actual == item['accepted_stat'], str(path)
    return dict(acceptance_sha256=sha(acceptance), stat_freeze_sha256=sha(freeze_path),
                original_NPZ_rehashed=False, accepted_stat_matches=True)


def run():
    source = CASE / 'source/ogbench'
    for name, expected in PINS.items():
        assert sha(source / name) == expected, name
    assert sha(OLD / 'selected_input.json') == SELECTED_SHA
    assert json.loads((OLD / 'selected_input.json').read_text())['selected_mode'] == 'prefetch1'
    data = verify_data()
    sys.path[:0] = [str(source / 'impls'), str(source)]
    spec = importlib.util.spec_from_file_location('_cube_formal_official_main', source / 'impls/main.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    official.flags.DEFINE_bool('defer_online_evaluation', False, 'Explicitly evaluate saved nodes after all training.')
    official.flags.DEFINE_bool('save_first_eval_checkpoint', False, 'Also save the original first evaluation node.')

    def entry(argv):
        from agents.gciql import get_config
        flags, config = official.FLAGS, official.FLAGS.agent
        expected = get_config()
        expected.alpha = 1.0
        assert config.to_dict() == expected.to_dict(), 'Only the official task alpha override is permitted'
        required = dict(seed=0, env_name='cube-triple-play-v0', train_steps=1000000,
            log_interval=5000, eval_interval=100000, save_interval=100000, eval_tasks=None,
            eval_episodes=50, eval_temperature=0., eval_gaussian=None, video_episodes=1,
            video_frame_skip=3, eval_on_cpu=1, restore_path=None, restore_epoch=None)
        assert all(getattr(flags, name) == value for name, value in required.items()), required
        assert flags.defer_online_evaluation and flags.save_first_eval_checkpoint
        assert any(device.platform == 'gpu' for device in official.jax.devices())
        assert official.jax.devices('cpu'), 'Official eval_on_cpu requires a visible CPU backend'
        state_dir = Path(os.environ['CUBE_FORMAL_RUN_DIR']).resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        write_new(state_dir / 'entry_binding.json', dict(source_sha256=PINS,
            adapter_sha256=ADAPTER_SHA, selected_profile_sha256=SELECTED_SHA, data=data,
            entry_sha256=sha(__file__), official_flags=official.get_flag_dict(),
            official_source_bytes_unmodified=True, declared_in_memory_AST_sites=2,
            formal_training=True, host_numpy_rng_separate_sidecar=True,
            wandb_mode='offline', state_dim=46, goal_dim=46, action_dim=5, input_mode='prefetch1'))
        with FormalInputBridge(official, CASE / 'data/official_state', state_dir) as bridge:
            code, proof = compile_deferred_main(official, (source / 'impls/main.py').read_bytes(),
                                               bridge.defer_evaluation_node)
            write_new(state_dir / 'entry_AST_proof.json', proof)
            official.main(argv)
            bridge.run_deferred_evaluations(code)

    official.app.run(entry)


if __name__ == '__main__':
    run()
