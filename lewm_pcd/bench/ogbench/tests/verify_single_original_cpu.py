"""Bounded CPU proof of Single setup/routing, with no fit or optimizer.

Use --vectors with the SHA-bound first/last episode action+observation NPZ
read from the historical data. Visual fixture columns are dummy fill values.
The production binding validator is replaced only for this diagnostic fixture;
this script does not establish production-data or model-training acceptance.
"""
import argparse
import ast
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import traceback
import types
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'third_party/stable-worldmodel'))
sys.dont_write_bytecode = True
VECTORS_SHA256 = '860178760b357c7471eb8b73604700b3a3e25fd40c08e3994e522dc9a69ad1a7'
ORIGINAL_SPLIT_SOURCE_SHA256 = '739da42fab8ce07e3fcedcf869548c32f89877077874ae0c8461764d4d65d526'


def verify(output, vectors, original_split_source):
    import h5py
    import hydra
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from bench.ogbench import lewm, single_original

    torch.set_num_threads(1)
    if hashlib.sha256(vectors.read_bytes()).hexdigest() != VECTORS_SHA256:
        raise ValueError('historical numeric-vector checksum differs')
    samples = np.load(vectors, allow_pickle=False)

    def digest(value):
        return hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).hexdigest()

    def exact(left, right):
        if isinstance(left, torch.Tensor):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError('tensor shape/dtype differs')
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                raise ValueError('comparison keys differ')
            for key in left:
                exact(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            if len(left) != len(right):
                raise ValueError('comparison lengths differ')
            for a, b in zip(left, right):
                exact(a, b)
        elif left != right:
            raise ValueError('comparison value differs')

    results = {}
    with lewm.pinned_native(single_original=True) as native:
        actual_split = native.spt.data.random_split
        split_source = Path(inspect.getsourcefile(actual_split))
        if hashlib.sha256(original_split_source.read_bytes()).hexdigest() != ORIGINAL_SPLIT_SOURCE_SHA256:
            raise ValueError('original 41134 SPT split source checksum differs')
        original_tree = ast.parse(original_split_source.read_text(encoding='utf8'))
        original_function = next(node for node in original_tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name == 'random_split')
        actual_function = ast.parse(textwrap.dedent(inspect.getsource(actual_split))).body[0]
        if ast.dump(original_function, include_attributes=False) != ast.dump(actual_function, include_attributes=False):
            raise ValueError('runtime SPT random_split function differs from original 41134 AST')
        manager_signature = str(inspect.signature(native.spt.Manager))
        actual_loader = native.swm.data.load_dataset
        actual_instantiate = hydra.utils.instantiate
        actual_normalizer = native.get_column_normalizer

        class IndexDataset:
            def __len__(self):
                return 1820000
            def __getitem__(self, index):
                return index

        # Use the actual SPT splitter and the original subtraction expression.
        # Remainder allocation can differ from a nominal integer 90/10 ratio.
        full_splits = []
        for _ in range(2):
            generator = torch.Generator().manual_seed(3072)
            split = actual_split(IndexDataset(), lengths=[.9, 1 - .9], generator=generator)
            full_splits.append(dict(counts=[len(part) for part in split],
                                    ids=[digest(part.indices) for part in split],
                                    generator=generator.get_state()))
        exact(full_splits[0], full_splits[1])
        full = dict(window_count=1820000, split_counts=full_splits[0]['counts'],
                    full_index_sha256=full_splits[0]['ids'],
                    shared_generator_state_sha256=hashlib.sha256(full_splits[0]['generator'].numpy().tobytes()).hexdigest(),
                    fractions=[.9, 1 - .9], original_spt_splitter=True)

        for mode in ('2d', '3d'):
            fixture = output / (mode + '_diagnostic_raw201.h5')
            lengths = samples[mode + '_ep_len']
            offsets = samples[mode + '_ep_offset']
            action = samples[mode + '_action']
            observation = samples[mode + '_observation']
            if action.shape != (402, 5) or observation.shape != (402, 28):
                raise ValueError('historical numeric-vector schema differs')
            if not np.array_equal(lengths, [201, 201]) or not np.array_equal(offsets, [0, 201]):
                raise ValueError('diagnostic raw201 episode indices differ')
            if not np.isnan(action[[200, 401]]).all():
                raise ValueError('historical terminal NaN actions were not preserved')
            with h5py.File(fixture, 'x', libver='latest') as handle:
                for key, data in (('ep_len', lengths), ('ep_offset', offsets),
                                  ('action', action), ('observation', observation)):
                    handle.create_dataset(key, data=data)
                if mode == '2d':
                    handle.create_dataset('pixels', shape=(402, 224, 224, 3), dtype='uint8', fillvalue=0)
                    handle.create_dataset('proprio', shape=(402, 1), dtype='float32', fillvalue=0)
                else:
                    handle.create_dataset('pcd_xyz', shape=(402, 1200, 3), dtype='float32', fillvalue=0)
                    handle.create_dataset('pcd_rgb', shape=(402, 1200, 3), dtype='uint8', fillvalue=0)
            binding = single_original.describe_single_original(mode, output)
            binding['dataset_path'] = str(fixture)
            source_cfg = single_original.load_single_recipe(mode)
            wrapped_cfg = lewm.build_config('single200', mode, output, workers=0, single_original=True)
            # Resource changes are explicit and shared by reference and wrapper.
            source_cfg['loader']['num_workers'] = source_cfg['num_workers'] = 0
            source_cfg['loader'].pop('persistent_workers', None)
            source_cfg['loader'].pop('prefetch_factor', None)
            for key in ('model', 'optimizer', 'loss', 'trainer', 'seed', 'train_split'):
                exact(source_cfg[key], wrapped_cfg[key])
            source_cfg['data']['dataset']['name'] = wrapped_cfg['data']['dataset']['name'] = str(fixture)
            exact(source_cfg['data'], wrapped_cfg['data'])
            initial_rng = torch.Generator().manual_seed(87123).get_state()

            def setup(cfg_dict, routed):
                record = dict(normalizers=[], normalized={}, manager_calls=0)
                cfg = OmegaConf.create(cfg_dict)

                def capture_loader(name, **kwargs):
                    dataset = actual_loader(name, **kwargs)
                    record['dataset'] = dataset
                    record['reader_class'] = type(dataset).__module__ + '.' + type(dataset).__name__
                    if type(dataset.clip_indices) is not list:
                        raise ValueError('original Python-list clip indices were replaced')
                    return dataset

                def normalizer(dataset, source, target):
                    record['normalizers'].append(source)
                    record['normalizer_window_counts'] = record.get('normalizer_window_counts', []) + [len(dataset)]
                    transform = actual_normalizer(dataset, source, target)
                    column = torch.from_numpy(np.array(dataset.get_col_data(source)))
                    record['normalized'][source] = transform({source: column})[target]
                    return transform

                def split(dataset, lengths, generator):
                    parts = actual_split(dataset, lengths=lengths, generator=generator)
                    record['split_ids'] = [list(part.indices) for part in parts]
                    record['split_generator'] = generator
                    record['after_split_rng'] = generator.get_state().clone()
                    return parts

                def instantiate(config, *args, **kwargs):
                    if config is cfg.model:
                        record['pre_model_global_rng'] = torch.get_rng_state().clone()
                        model = torch.nn.Linear(4, 4)
                        record['model_init'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
                        return model
                    return actual_instantiate(config, *args, **kwargs)

                def data_module(**kwargs):
                    record['loaders'] = kwargs
                    return types.SimpleNamespace(**kwargs)

                def manager(**kwargs):
                    if 'seed' in kwargs:
                        raise ValueError('Single Manager seed argument must remain omitted')
                    record['manager_seed_omitted'] = True
                    def no_fit():
                        record['manager_calls'] += 1
                    return no_fit

                run_home = output / (mode + ('_wrapped' if routed else '_reference'))
                run_home.mkdir()
                with patch.dict(os.environ, {'STABLEWM_HOME': str(run_home)}), \
                     patch.object(native.swm.data, 'load_dataset', side_effect=capture_loader), \
                     patch.object(native, 'get_column_normalizer', side_effect=normalizer), \
                     patch.object(native.spt.data, 'random_split', side_effect=split), \
                     patch.object(native.spt.data, 'DataModule', side_effect=data_module), \
                     patch.object(hydra.utils, 'instantiate', side_effect=instantiate), \
                     patch.object(native.spt, 'Module', side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)), \
                     patch.object(native.spt, 'Manager', side_effect=manager), \
                     patch.object(native.pl, 'Trainer', side_effect=lambda **kwargs: types.SimpleNamespace(**kwargs)), \
                     patch.object(single_original, 'describe_single_original', return_value=binding), \
                     patch.object(single_original, 'validate_single_original_binding', return_value=binding):
                    torch.set_rng_state(initial_rng.clone())
                    with lewm.route_datasets(native, binding, output) if routed else nullcontext():
                        native.run.__wrapped__(cfg)
                record['post_setup_global_rng'] = torch.get_rng_state().clone()
                train, val = record['loaders']['train'], record['loaders']['val']
                if train.generator is not record['split_generator'] or train.sampler.generator is not train.generator:
                    raise ValueError('split and shuffled train loader lost the original shared generator')
                record['sampler_order'] = list(train.sampler)
                record['post_sampler_rng'] = train.generator.get_state().clone()
                record['validation_count'] = len(val.dataset)
                dataset = record['dataset']
                record['clip_indices'] = dataset.clip_indices[:]
                record['first_window'] = dataset[0]
                record['last_window'] = dataset[len(dataset) - 1]
                if record['clip_indices'][-1] != (1, 181):
                    raise ValueError('last raw201 window was dropped or shifted')
                if record['last_window']['action'].shape != (4, 25) or int(torch.isnan(record['last_window']['action']).sum()) != 5:
                    raise ValueError('terminal NaN action/window semantics differ')
                if record['normalizers'] != ['action', 'observation'] or record['normalizer_window_counts'] != [364, 364]:
                    raise ValueError('full-source action/observation normalizers did not precede splitting')
                if dataset.h5_file is not None:
                    dataset.h5_file.close()
                    dataset.h5_file = None
                for key in ('dataset', 'loaders', 'split_generator'):
                    record.pop(key)
                return record

            reference, wrapped = setup(source_cfg, False), setup(wrapped_cfg, True)
            exact(reference, wrapped)
            np.savez(output / (mode + '_full_diagnostic_split_indices.npz'),
                     train=np.asarray(wrapped['split_ids'][0], dtype=np.int64),
                     validation=np.asarray(wrapped['split_ids'][1], dtype=np.int64),
                     train_sampler=np.asarray(wrapped['sampler_order'], dtype=np.int64))
            results[mode] = dict(normalizers=['action', 'observation'],
                normalizers_before_split_exact=True, full_diagnostic_split_ids_exact=True,
                split_counts=[len(ids) for ids in wrapped['split_ids']],
                split_index_sha256=[digest(ids) for ids in wrapped['split_ids']],
                shared_split_shuffle_generator_exact=True, sampler_order_and_post_rng_exact=True,
                model_init_and_global_rng_exact=True, manager_seed_argument_omitted=True,
                original_reader_class=wrapped['reader_class'], original_list_clip_indices=True,
                windows_per_episode=182, last_start=181, terminal_nan_action_count=5,
                first_and_terminal_window_exact=True, fixture_graphics='dummy_fill_values',
                real_numeric_source_row_ids=samples[mode + '_original_row_ids'].tolist())
    return dict(device='cpu', torch_version=torch.__version__, source_train_sha256=lewm.ORIGINAL_SHA256,
        numeric_vectors_sha256=VECTORS_SHA256, random_split_source=str(split_source),
        original_split_source_sha256=ORIGINAL_SPLIT_SOURCE_SHA256, random_split_original_function_ast_exact=True,
        random_split_source_sha256=hashlib.sha256(split_source.read_bytes()).hexdigest(),
        manager_runtime_signature=manager_signature, full_source_window_split=full, modes=results,
        fit_calls=0, optimizer_updates=0, model_training=False, production_binding_validator_bypassed_for_fixture=True,
        production_data_verified=False, cuda_bf16_verified=False, complete_training_epochs=0,
        formal_acceptance_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vectors', type=Path, required=True)
    parser.add_argument('--original-split-source', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='single_original_cpu_running', started_at=datetime.now(timezone.utc).isoformat(),
                 argv=sys.argv[:], actual_exit_code=None)
    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = output / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf8')
        temporary.replace(output / 'status.json')
    save()
    with (output / 'stdout_stderr.log').open('x', encoding='utf8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                result = verify(output, args.vectors.resolve(), args.original_split_source.resolve())
                (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
                state.update(stage='single_original_cpu_verified', actual_exit_code=0)
            except BaseException:
                state.update(stage='single_original_cpu_failed', actual_exit_code=1, error=traceback.format_exc())
                traceback.print_exc()
    state['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    (output / 'actual_exit_code').write_text(str(state['actual_exit_code']) + '\n', encoding='utf8')
    return state['actual_exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
